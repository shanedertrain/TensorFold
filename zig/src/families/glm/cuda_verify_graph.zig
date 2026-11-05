//! TF_GLM_VERIFY_GRAPH=1: a verify window's forward replayed from CUDA graphs, the same kernels in the same order.
//!
//! One verify call issues some 1,600 kernels, nearly all small and dependent (the hyper-connection glue, the KDA
//! chain, norms, the router, the K-slice reduces), so the GPU idles a few microseconds between them. A window's
//! kernels take every pointer from the engine's fixed buffers and read the position from the device, so a call is a
//! pure function of its key: the rows, the KDA state parity, whether the MLA layers run the sparse pass, and the
//! sequence (its generation and its index rings' base). The MLA attention and the token selection size their grids
//! from the host position, so they stay eager: the forward is cut into segments around them (one cut an MLA layer),
//! each segment a graph.
//!
//! A key runs eagerly the first time (a one-off window never pays a capture, and lazily grown scratch is allocated),
//! is recorded the second time (a dry pass: the segments captured, the eager regions skipped, nothing executed),
//! instantiated, agreed by every rank, and then replayed. A capture that fails on any rank turns graphs off on every
//! rank, because the ranks' gathers must stay in step. Bits cannot change: a graph holds exactly the launches the
//! eager pass would issue, with their arguments.

const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const prof_mod = @import("cuda_prof.zig");
const fw = @import("cuda_forward.zig");
const dsa_block = @import("cuda_dsa_block.zig");

const quiet = builtin.is_test;

/// TF_GLM_VERIFY_GRAPH=1 asks for it (every rank must agree; the default is off until it is measured).
pub fn enabled() bool {
    const v = std.c.getenv("TF_GLM_VERIFY_GRAPH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Graphs held at most (TF_GLM_VERIFY_GRAPH_MAX, default 24): rows 1..8 at both parities is 16 a sequence.
pub fn maxGraphs() u32 {
    const v = std.c.getenv("TF_GLM_VERIFY_GRAPH_MAX") orelse return 24;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 24;
}

/// Everything a window's launches bake in that may differ between two calls; the rest is read from the device.
pub const Key = struct {
    rows: u8,
    parity: u8, // st.cur[0]: which KDA state copy is read and which is written (commit flips every layer's)
    sparse: bool, // the MLA layers' rows past the dense limit take the indexer's query projection
    gen: u64, // the sequence's number (Seq.gen): a freed slot's graphs never match the sequence after it
    base: u64, // st.base: a ring rebase moves the index caches' pointers

    /// The key of a window of `rows` on `f`'s one rank, or null when the window cannot be graphed.
    pub fn of(f: *const fw.Forward, rows: usize, gen: u64) ?Key {
        if (f.ranks.len != 1 or f.exchange != .nccl or f.images != null or f.pieces != null) return null;
        const r = f.ranks[0];
        if (r.segs.len > 0 or r.b.prefill or rows == 0 or rows > fw.window_rows) return null;
        const st = r.st;
        if (st.cur.len == 0) return null;
        const parity = st.cur[0];
        for (st.cur) |x| if (x != parity) return null; // out of step: commit refuses it; this window stays eager
        var sparse = false;
        for (r.w.layers) |l| if (l.mixer == .mla) {
            sparse = dsa_block.sparseRows(l.mixer.mla, st.pos, rows);
            break;
        };
        return .{ .rows = @intCast(rows), .parity = parity, .sparse = sparse, .gen = gen, .base = st.base };
    }
};

/// One captured segment, as the engine holds it.
pub const Exec = *anyopaque;

/// Capture a segment of the stream into an executable, launch it, free it: a fake engine runs the policy in host tests.
pub const Engine = struct {
    ctx: *anyopaque,
    begin: *const fn (ctx: *anyopaque, stream: cuda.Stream) anyerror!void,
    /// Ends the capture `begin` started and instantiates it (uploaded); the stream is out of capture on every path.
    end: *const fn (ctx: *anyopaque, stream: cuda.Stream) anyerror!Exec,
    /// Ends and drops a capture a failed body left open (none open: nothing).
    abort: *const fn (ctx: *anyopaque, stream: cuda.Stream) void,
    launch: *const fn (ctx: *anyopaque, exec: Exec, stream: cuda.Stream) anyerror!void,
    free: *const fn (ctx: *anyopaque, exec: Exec) void,
};

/// Every rank's verdict AND-ed, outside any capture; called on a miss alone, in the same order on every rank.
pub const Agree = struct {
    ctx: *anyopaque,
    all: *const fn (ctx: *anyopaque, ok: bool) anyerror!bool,
};

/// The kernels of one window, through `s` (null: eagerly); the same launches in the same order every time.
pub const Body = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, s: ?*Session) anyerror!void,
};

pub const Outcome = enum { eager, captured, replayed };

/// A key's graphs: its segments in forward order, and the profiler marks each one records (as event nodes).
const Held = struct {
    execs: std.ArrayList(Exec) = .empty,
    marks: std.ArrayList(prof_mod.Section) = .empty, // every mark of the call in order
    counts: std.ArrayList(u16) = .empty, // marks a segment
    used: u64 = 0,

    fn deinit(h: *Held, gpa: std.mem.Allocator, engine: Engine) void {
        for (h.execs.items) |x| engine.free(engine.ctx, x);
        h.execs.deinit(gpa);
        h.marks.deinit(gpa);
        h.counts.deinit(gpa);
        h.* = undefined;
    }
};

pub const Mode = enum { record, replay };

/// What the forward sees: segments opened and closed around its graphable spans, eager regions between them.
pub const Session = struct {
    mode: Mode,
    engine: Engine,
    stream: cuda.Stream,
    gpa: std.mem.Allocator,
    held: *Held,
    prof: ?*prof_mod.Prof,
    at: usize = 0, // segments opened
    mark0: usize = 0, // record: the profiler's marks when the open segment began
    is_open: bool = false,

    /// A graphable span begins: true when the body must issue its kernels (into the recording); false: replayed.
    pub fn open(s: *Session) !bool {
        if (s.is_open) return error.GraphSpanOpen;
        switch (s.mode) {
            .record => {
                try s.engine.begin(s.engine.ctx, s.stream);
                s.is_open = true;
                if (s.prof) |p| {
                    p.capture = true;
                    s.mark0 = p.n;
                }
                return true;
            },
            .replay => {
                if (s.at >= s.held.execs.items.len) return error.GraphSegments;
                try s.engine.launch(s.engine.ctx, s.held.execs.items[s.at], s.stream);
                if (s.prof) |p| { // the graph's event nodes record the marks; the host notes their sections
                    var off: usize = 0;
                    for (s.held.counts.items[0..s.at]) |c| off += c;
                    for (s.held.marks.items[off .. off + s.held.counts.items[s.at]]) |m| p.note(m);
                }
                s.at += 1;
                return false;
            },
        }
    }

    /// The span ends: recorded, instantiated and kept (a replay launched it at `open`).
    pub fn close(s: *Session) !void {
        if (s.mode != .record) return;
        if (!s.is_open) return error.GraphSpanClosed;
        s.is_open = false;
        if (s.prof) |p| p.capture = false;
        const exec = try s.engine.end(s.engine.ctx, s.stream);
        errdefer s.engine.free(s.engine.ctx, exec);
        try s.held.execs.append(s.gpa, exec);
        var marks: usize = 0;
        if (s.prof) |p| {
            marks = p.n - s.mark0;
            try s.held.marks.appendSlice(s.gpa, p.ends[s.mark0..p.n]);
        }
        try s.held.counts.append(s.gpa, @intCast(marks));
        s.at += 1;
    }

    /// Whether the kernels between two spans (grids from the host position) are issued now: not in a recording pass,
    /// which issues nothing and leaves them to its replay.
    pub fn eager(s: *const Session) bool {
        return s.mode == .replay;
    }
};

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    engine: Engine,
    max: u32,
    agree: ?Agree = null,
    cuda_engine: CudaEngine = undefined, // `engine`'s context when made by `create`
    held: std.AutoHashMapUnmanaged(Key, *Held) = .empty,
    /// Keys seen once: a key runs eagerly first and is captured the second time.
    seen: std.AutoHashMapUnmanaged(Key, void) = .empty,
    tick: u64 = 0,
    off: bool = false, // a capture failed or a rank refused: every window eager from here (logged once)
    stats: struct { captured: u64 = 0, replayed: u64 = 0, eager: u64 = 0, evicted: u64 = 0 } = .{},

    pub fn init(gpa: std.mem.Allocator, engine: Engine, max: u32) Graphs {
        return .{ .gpa = gpa, .engine = engine, .max = @max(max, 1) };
    }

    /// The graphs on `d`'s driver, on the heap so the engine's context stays put.
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, max: u32) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .gpa = gpa, .engine = undefined, .max = @max(max, 1), .cuda_engine = .{ .d = d } };
        g.engine = g.cuda_engine.engine();
        return g;
    }

    pub fn destroy(g: *Graphs) void {
        const gpa = g.gpa;
        if (!quiet) std.log.info("glm verify graphs: {d} captured, {d} replayed, {d} eager, {d} held, {d} evicted{s}", .{ g.stats.captured, g.stats.replayed, g.stats.eager, g.held.count(), g.stats.evicted, if (g.off) " (off after a refusal)" else "" });
        g.deinit();
        gpa.destroy(g);
    }

    pub fn deinit(g: *Graphs) void {
        g.dropAll();
        g.held.deinit(g.gpa);
        g.seen.deinit(g.gpa);
    }

    pub fn count(g: *const Graphs) usize {
        return g.held.count();
    }

    pub fn has(g: *const Graphs, key: Key) bool {
        return g.held.contains(key);
    }

    fn dropAll(g: *Graphs) void {
        var it = g.held.valueIterator();
        while (it.next()) |h| {
            h.*.deinit(g.gpa, g.engine);
            g.gpa.destroy(h.*);
        }
        g.held.clearRetainingCapacity();
    }

    fn evictOldest(g: *Graphs) void {
        var oldest: ?Key = null;
        var when: u64 = std.math.maxInt(u64);
        var it = g.held.iterator();
        while (it.next()) |e| if (e.value_ptr.*.used < when) {
            when = e.value_ptr.*.used;
            oldest = e.key_ptr.*;
        };
        const k = oldest orelse return;
        const h = g.held.fetchRemove(k).?.value;
        h.deinit(g.gpa, g.engine);
        g.gpa.destroy(h);
        g.stats.evicted += 1;
    }

    fn agreed(g: *Graphs, ok: bool) !bool {
        const a = g.agree orelse return ok;
        return a.all(a.ctx, ok);
    }

    fn refuse(g: *Graphs, why: []const u8, key: Key) void {
        if (!quiet) std.log.warn("glm verify graphs off: {s} (rows {d}, parity {d}); every window eager from here", .{ why, key.rows, key.parity });
        g.off = true;
        g.dropAll();
    }

    /// One window under `key`: eager at first sight, recorded then replayed at the second, replayed after.
    pub fn run(g: *Graphs, key: Key, stream: cuda.Stream, prof: ?*prof_mod.Prof, body: Body) !Outcome {
        if (g.off) {
            g.stats.eager += 1;
            try body.run(body.ctx, null);
            return .eager;
        }
        g.tick += 1;
        if (g.held.getPtr(key)) |hp| {
            const h = hp.*;
            h.used = g.tick;
            try g.replay(h, stream, prof, body);
            g.stats.replayed += 1;
            return .replayed;
        }
        if (!g.seen.contains(key)) {
            if (g.seen.count() >= 4 * g.max) g.seen.clearRetainingCapacity();
            try g.seen.put(g.gpa, key, {});
            g.stats.eager += 1;
            try body.run(body.ctx, null);
            return .eager;
        }
        // the second sight: a dry recording pass (nothing runs), every rank's verdict, then the replay
        while (g.held.count() >= g.max) g.evictOldest();
        try g.held.ensureUnusedCapacity(g.gpa, 1);
        const h = try g.gpa.create(Held);
        h.* = .{};
        var ok = true;
        {
            var s: Session = .{ .mode = .record, .engine = g.engine, .stream = stream, .gpa = g.gpa, .held = h, .prof = prof };
            body.run(body.ctx, &s) catch |err| {
                g.engine.abort(g.engine.ctx, stream);
                if (prof) |p| p.capture = false;
                if (!quiet) std.log.warn("glm verify graphs: recording failed ({t}) at rows {d}", .{ err, key.rows });
                ok = false;
            };
            if (ok and (s.is_open or h.execs.items.len == 0)) ok = false; // a span left open, or no span at all
        }
        const all = g.agreed(ok) catch |err| {
            h.deinit(g.gpa, g.engine);
            g.gpa.destroy(h);
            return err;
        };
        if (!all) {
            h.deinit(g.gpa, g.engine);
            g.gpa.destroy(h);
            g.refuse(if (ok) "another rank's capture failed" else "this rank's capture failed", key);
            if (prof) |p| try p.begin(stream, p.label); // the dry pass marked nothing that ran
            g.stats.eager += 1;
            try body.run(body.ctx, null);
            return .eager;
        }
        g.held.putAssumeCapacityNoClobber(key, h);
        h.used = g.tick;
        g.stats.captured += 1;
        if (!quiet and (g.stats.captured == 1 or g.stats.captured % 16 == 0)) std.log.info("glm verify graphs: captured {d} (rows {d}, parity {d}, {d} segments), {d} held", .{ g.stats.captured, key.rows, key.parity, h.execs.items.len, g.held.count() });
        if (prof) |p| try p.begin(stream, p.label); // the replay's marks, not the dry pass's
        try g.replay(h, stream, prof, body);
        return .captured;
    }

    fn replay(g: *Graphs, h: *Held, stream: cuda.Stream, prof: ?*prof_mod.Prof, body: Body) !void {
        var s: Session = .{ .mode = .replay, .engine = g.engine, .stream = stream, .gpa = g.gpa, .held = h, .prof = prof };
        try body.run(body.ctx, &s);
        if (s.at != h.execs.items.len) return error.GraphSegments; // the body cut fewer spans than it recorded
    }

    /// Engine.verify's forward of `ids` on `f` (one rank, a window): through the key's graphs when it has a key.
    pub fn forward(g: *Graphs, f: *fw.Forward, ids: []const i32, gen: u64) !void {
        try f.prepare(ids);
        const R = ids.len;
        const key = Key.of(f, R, gen) orelse return f.body(R);
        var a: Adapter = .{ .f = f, .rows = R };
        _ = try g.run(key, f.stream, f.prof, .{ .ctx = &a, .run = Adapter.run });
    }

    const Adapter = struct {
        f: *fw.Forward,
        rows: usize,

        fn run(ctx: *anyopaque, s: ?*Session) anyerror!void {
            const a: *Adapter = @ptrCast(@alignCast(ctx));
            a.f.graph = s;
            defer a.f.graph = null;
            try a.f.body(a.rows);
        }
    };
};

/// The driver's engine: thread-local stream capture (other threads' CUDA calls stay out), instantiate, upload, launch.
pub const CudaEngine = struct {
    d: *const cuda.Driver,

    pub fn engine(e: *CudaEngine) Engine {
        return .{ .ctx = e, .begin = begin, .end = end, .abort = abort, .launch = launch, .free = free };
    }

    const Held2 = struct { exec: cuda.graph.Exec };

    fn begin(_: *anyopaque, stream: cuda.Stream) anyerror!void {
        try cuda.graph.beginCapture(stream, .thread_local);
    }

    fn end(_: *anyopaque, stream: cuda.Stream) anyerror!Exec {
        var g = try cuda.graph.endCapture(stream);
        defer g.deinit();
        const h = try std.heap.c_allocator.create(Held2);
        errdefer std.heap.c_allocator.destroy(h);
        h.exec = try g.instantiate();
        errdefer h.exec.deinit();
        try h.exec.upload(stream);
        return h;
    }

    fn abort(_: *anyopaque, stream: cuda.Stream) void {
        if ((cuda.graph.captureStatus(stream) catch .none) == .none) return;
        if (cuda.graph.endCapture(stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
    }

    fn launch(_: *anyopaque, exec: Exec, stream: cuda.Stream) anyerror!void {
        const h: *Held2 = @ptrCast(@alignCast(exec));
        try h.exec.launchOn(stream);
    }

    fn free(_: *anyopaque, exec: Exec) void {
        const h: *Held2 = @ptrCast(@alignCast(exec));
        h.exec.deinit();
        std.heap.c_allocator.destroy(h);
    }
};

// -- host tests ---------------------------------------------------------------------------------------------------

const t = std.testing;
const no_stream: cuda.Stream = .{ .d = undefined, .handle = null };

/// A fake engine that counts: captures hand out slots, launches and frees are tallied, a capture may be made to fail.
const Fake = struct {
    begins: u32 = 0,
    ends: u32 = 0,
    aborts: u32 = 0,
    launches: u32 = 0,
    live: u32 = 0,
    fail_end: bool = false,
    slots: [64]u8 = undefined,

    fn engine(f: *Fake) Engine {
        return .{ .ctx = f, .begin = begin, .end = end, .abort = abort, .launch = launch, .free = free };
    }
    fn begin(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.begins += 1;
    }
    fn end(ctx: *anyopaque, _: cuda.Stream) anyerror!Exec {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_end) return error.GraphNotPermitted;
        f.ends += 1;
        f.live += 1;
        return &f.slots[f.ends % f.slots.len];
    }
    fn abort(ctx: *anyopaque, _: cuda.Stream) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.aborts += 1;
    }
    fn launch(ctx: *anyopaque, _: Exec, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.launches += 1;
    }
    fn free(ctx: *anyopaque, _: Exec) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.live -= 1;
    }
};

/// A body of `spans` graphable spans with an eager region between each pair; counts what it issued and where.
const Spans = struct {
    spans: usize = 3,
    issued: u32 = 0, // kernels issued eagerly or into a recording
    eager_regions: u32 = 0,
    skipped: u32 = 0, // spans a replay launched instead

    fn body(b: *Spans) Body {
        return .{ .ctx = b, .run = run };
    }
    fn run(ctx: *anyopaque, s: ?*Session) anyerror!void {
        const b: *Spans = @ptrCast(@alignCast(ctx));
        const f: fw.Forward = .{ .d = undefined, .m = undefined, .stream = no_stream, .ranks = &.{}, .exchange = .local, .graph = s };
        for (0..b.spans) |i| {
            if (i > 0) {
                if (f.eagerRegion()) b.eager_regions += 1;
            }
            if (try f.spanOpen()) b.issued += 1 else b.skipped += 1;
            try f.spanClose();
        }
    }
};

fn keyOf(rows: u8, parity: u8) Key {
    return .{ .rows = rows, .parity = parity, .sparse = false, .gen = 1, .base = 0 };
}

test "GLM verify graphs: eager at first sight, recorded then replayed at the second, replayed after; LRU at the cap" {
    var f: Fake = .{};
    var b: Spans = .{};
    var g = Graphs.init(t.allocator, f.engine(), 2);
    defer g.deinit();
    const k = keyOf(2, 0);
    try t.expectEqual(Outcome.eager, try g.run(k, no_stream, null, b.body()));
    try t.expectEqual([3]u32{ 3, 2, 0 }, [3]u32{ b.issued, b.eager_regions, b.skipped });
    try t.expectEqual(Outcome.captured, try g.run(k, no_stream, null, b.body()));
    // the dry pass issued 3 spans into captures and no eager region; the replay launched 3 graphs and 2 regions
    try t.expectEqual([3]u32{ 6, 4, 3 }, [3]u32{ b.issued, b.eager_regions, b.skipped });
    try t.expectEqual([4]u32{ 3, 3, 3, 3 }, [4]u32{ f.begins, f.ends, f.launches, f.live });
    try t.expectEqual(Outcome.replayed, try g.run(k, no_stream, null, b.body()));
    try t.expectEqual([3]u32{ 6, 6, 6 }, [3]u32{ b.issued, b.eager_regions, b.skipped });
    try t.expectEqual(@as(u32, 6), f.launches);
    // other keys: another parity and another row count each get their own graphs, the least recent go at the cap
    const k1 = keyOf(2, 1);
    const k2 = keyOf(3, 0);
    _ = try g.run(k1, no_stream, null, b.body());
    try t.expectEqual(Outcome.captured, try g.run(k1, no_stream, null, b.body()));
    try t.expectEqual(@as(usize, 2), g.count());
    _ = try g.run(k2, no_stream, null, b.body());
    try t.expectEqual(Outcome.captured, try g.run(k2, no_stream, null, b.body()));
    try t.expectEqual(@as(usize, 2), g.count());
    try t.expect(!g.has(k) and g.has(k1) and g.has(k2));
    try t.expectEqual(@as(u64, 1), g.stats.evicted);
    try t.expectEqual(@as(u32, 6), f.live);
}

test "GLM verify graphs: a failed capture or a rank's refusal turns graphs off, the window runs eagerly, nothing leaks" {
    var f: Fake = .{};
    var b: Spans = .{};
    var g = Graphs.init(t.allocator, f.engine(), 8);
    defer g.deinit();
    const k = keyOf(1, 0);
    _ = try g.run(k, no_stream, null, b.body());
    f.fail_end = true;
    try t.expectEqual(Outcome.eager, try g.run(k, no_stream, null, b.body()));
    try t.expect(g.off);
    try t.expectEqual(@as(u32, 0), f.live);
    try t.expectEqual(@as(u32, 1), f.aborts); // the failed body's capture ended
    try t.expectEqual(@as(u32, 0), f.launches);
    f.fail_end = false;
    try t.expectEqual(Outcome.eager, try g.run(k, no_stream, null, b.body())); // off stays off
    try t.expectEqual(@as(u32, 1), f.begins); // no further recording
    // another rank's refusal: this rank's good graphs go too
    var f2: Fake = .{};
    var g2 = Graphs.init(t.allocator, f2.engine(), 8);
    defer g2.deinit();
    const No = struct {
        fn all(_: *anyopaque, _: bool) anyerror!bool {
            return false;
        }
    };
    var ctx: u8 = 0;
    g2.agree = .{ .ctx = &ctx, .all = No.all };
    _ = try g2.run(k, no_stream, null, b.body());
    try t.expectEqual(Outcome.eager, try g2.run(k, no_stream, null, b.body()));
    try t.expect(g2.off);
    try t.expectEqual(@as(u32, 0), f2.live);
    try t.expectEqual(@as(u32, 0), f2.launches);
}

test "GLM verify graphs: the key tells rows, parity, the sparse pass, the sequence and its ring base apart" {
    const a = keyOf(2, 0);
    try t.expect(std.meta.eql(a, keyOf(2, 0)));
    try t.expect(!std.meta.eql(a, keyOf(2, 1)));
    try t.expect(!std.meta.eql(a, keyOf(3, 0)));
    var b = a;
    b.sparse = true;
    try t.expect(!std.meta.eql(a, b));
    b = a;
    b.gen = 2;
    try t.expect(!std.meta.eql(a, b));
    b = a;
    b.base = 4096;
    try t.expect(!std.meta.eql(a, b));
}

/// A driver whose launches land in the open capture or on the "device", whose graphs replay their launches.
const Rec = struct {
    var issued: [256]u64 = undefined;
    var n: usize = 0;
    var direct: usize = 0; // cuLaunchKernel calls outside a capture
    var capturing = false;
    var graphs: [16][64]u64 = undefined;
    var glen: [16]usize = undefined;
    var ngraphs: usize = 0;
    var cur: usize = 0;

    fn reset() void {
        n = 0;
        direct = 0;
        capturing = false;
        ngraphs = 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const v = (@as(u64, @intFromPtr(f)) << 32) | gx;
        if (capturing) {
            graphs[cur][glen[cur]] = v;
            glen[cur] += 1;
        } else {
            issued[n] = v;
            n += 1;
            direct += 1;
        }
        return cuda.abi.success;
    }
    fn begin(_: cuda.abi.Stream, _: cuda.abi.CaptureMode) callconv(.c) cuda.abi.Result {
        capturing = true;
        cur = ngraphs;
        glen[cur] = 0;
        ngraphs += 1;
        return cuda.abi.success;
    }
    fn end(_: cuda.abi.Stream, out: *cuda.abi.Graph) callconv(.c) cuda.abi.Result {
        capturing = false;
        out.* = @ptrFromInt(cur + 1);
        return cuda.abi.success;
    }
    fn status(_: cuda.abi.Stream, s: *cuda.abi.CaptureStatus) callconv(.c) cuda.abi.Result {
        s.* = if (capturing) .active else .none;
        return cuda.abi.success;
    }
    fn instantiate(e: *cuda.abi.GraphExec, g: cuda.abi.Graph, _: u64) callconv(.c) cuda.abi.Result {
        e.* = @ptrFromInt(@intFromPtr(g));
        return cuda.abi.success;
    }
    fn upload(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn glaunch(e: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const i = @intFromPtr(e) - 1;
        for (graphs[i][0..glen[i]]) |v| {
            issued[n] = v;
            n += 1;
        }
        return cuda.abi.success;
    }
    fn gdestroy(_: cuda.abi.Graph) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn edestroy(_: cuda.abi.GraphExec) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn errName(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = "fixture";
        return cuda.abi.success;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = glaunch;
        d.api.cuGraphDestroy = gdestroy;
        d.api.cuGraphExecDestroy = edestroy;
        d.api.cuGetErrorName = errName;
        d.api.cuGetErrorString = errName;
        return d;
    }
};

/// A forward's shape in miniature: an embedding, four layers of which two are "MLA" (attention eager between two
/// spans, its grid from the position), a head; every launch through cuda.launch.launch on the recording driver.
const Synth = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    pos: u32 = 7,

    fn body(x: *Synth) Body {
        return .{ .ctx = x, .run = run };
    }
    fn go(x: *Synth, id: usize, gx: u32) !void {
        var a: cuda.Args = .{};
        a.add(@as(u64, 0x1000));
        try cuda.launch.launch(.{ .d = x.d, .handle = @ptrFromInt(id) }, .{ .grid = .{ .x = gx }, .block = .{ .x = 32 } }, x.stream, &a);
    }
    fn run(ctx: *anyopaque, s: ?*Session) anyerror!void {
        const x: *Synth = @ptrCast(@alignCast(ctx));
        const f: fw.Forward = .{ .d = x.d, .m = undefined, .stream = x.stream, .ranks = &.{}, .exchange = .local, .graph = s };
        var open = try f.spanOpen();
        if (open) try x.go(1, 4); // embed
        for (0..4) |j| {
            if (open) try x.go(10 + j, 2); // hc_pre and the mixer's head
            if (j % 2 == 1) { // an MLA layer: its attention between two spans
                try f.spanClose();
                if (f.eagerRegion()) try x.go(50 + j, (x.pos + 511) / 512);
                open = try f.spanOpen();
                if (open) try x.go(60 + j, 2); // expand_v and o_proj
            }
            if (open) try x.go(20 + j, 3); // the gather, the glue and the ffn
        }
        if (open) try x.go(99, 1); // the head
        try f.spanClose();
    }
};

test "GLM verify graphs: a replay issues the eager pass's launch list, the position-bound kernels eagerly between graphs" {
    const d = Rec.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var e: CudaEngine = .{ .d = &d };
    var g = Graphs.init(t.allocator, e.engine(), 8);
    defer g.deinit();
    var x: Synth = .{ .d = &d, .stream = stream };
    const k = keyOf(2, 0);
    // the eager pass: every launch direct
    Rec.reset();
    try t.expectEqual(Outcome.eager, try g.run(k, stream, null, x.body()));
    var eager: [256]u64 = undefined;
    const len = Rec.n;
    @memcpy(eager[0..len], Rec.issued[0..len]);
    try t.expectEqual(@as(usize, 14), len);
    try t.expectEqual(len, Rec.direct);
    // the second sight: three segments recorded (nothing reaches the device), then replayed around the two eager kernels
    Rec.reset();
    try t.expectEqual(Outcome.captured, try g.run(k, stream, null, x.body()));
    try t.expectEqual(@as(usize, 3), Rec.ngraphs);
    try t.expectEqual(len, Rec.n);
    try t.expectEqualSlices(u64, eager[0..len], Rec.issued[0..len]);
    try t.expectEqual(@as(usize, 2), Rec.direct);
    // a replay: the same list again, only the attention launched directly
    Rec.reset();
    try t.expectEqual(Outcome.replayed, try g.run(k, stream, null, x.body()));
    try t.expectEqual(@as(usize, 0), Rec.ngraphs);
    try t.expectEqualSlices(u64, eager[0..len], Rec.issued[0..len]);
    try t.expectEqual(@as(usize, 2), Rec.direct);
    // the position moved: the eager kernels take their new grid, the graphs are untouched
    x.pos = 1500;
    Rec.reset();
    try t.expectEqual(Outcome.replayed, try g.run(k, stream, null, x.body()));
    try t.expect(!std.mem.eql(u64, eager[0..len], Rec.issued[0..len]));
    var moved: usize = 0;
    for (eager[0..len], Rec.issued[0..len]) |a, b| if (a != b) {
        moved += 1;
        try t.expectEqual(a >> 32, b >> 32); // the same kernel, another grid
        try t.expectEqual(@as(u64, 3), b & 0xffff_ffff);
    };
    try t.expectEqual(@as(usize, 2), moved);
    try t.expect(!(cuda.graph.captureStatus(stream) catch .active == .active)); // no capture left open
}

/// Events that count records: plain ones and external (event node) ones apart.
const Ev = struct {
    var live: usize = 0;
    var plain: usize = 0;
    var external: usize = 0;
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        live += 1;
        out.* = @ptrFromInt(live);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        plain += 1;
        return 0;
    }
    fn recordFlags(_: cuda.abi.Event, _: cuda.abi.Stream, flags: c_uint) callconv(.c) c_int {
        if (flags == cuda.abi.event_record_external) external += 1 else plain += 1;
        return 0;
    }
};

test "GLM verify graphs: a recording's profiler marks become event nodes, a replay notes their sections without recording" {
    var d: cuda.Driver = undefined;
    d.api.cuEventCreate = Ev.create;
    d.api.cuEventDestroy_v2 = Ev.destroy;
    d.api.cuEventRecord = Ev.record;
    d.api.cuEventRecordWithFlags = Ev.recordFlags;
    Ev.live = 0;
    Ev.plain = 0;
    Ev.external = 0;
    const p = try prof_mod.Prof.init(&d, t.io);
    defer p.deinit();
    var f: Fake = .{};
    var h: Held = .{};
    defer h.deinit(t.allocator, f.engine());
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    // the recording: two spans, two marks in the first and one in the second, every mark an external record
    try p.begin(stream, "verify 2 rows");
    try t.expectEqual(@as(usize, 1), Ev.plain);
    var s: Session = .{ .mode = .record, .engine = f.engine(), .stream = stream, .gpa = t.allocator, .held = &h, .prof = p };
    try t.expect(try s.open());
    p.mark(stream, .kda);
    p.mark(stream, .gather_a);
    try s.close();
    try t.expect(try s.open());
    p.mark(stream, .head);
    try s.close();
    try t.expectEqual(@as(usize, 3), Ev.external);
    try t.expectEqual(@as(usize, 1), Ev.plain);
    try t.expect(!p.capture);
    try t.expectEqualSlices(u16, &.{ 2, 1 }, h.counts.items);
    try t.expectEqualSlices(prof_mod.Section, &.{ .kda, .gather_a, .head }, h.marks.items);
    // the replay: the host notes the same sections in order, records nothing itself
    try p.begin(stream, "verify 2 rows");
    try t.expectEqual(@as(usize, 2), Ev.plain);
    var r: Session = .{ .mode = .replay, .engine = f.engine(), .stream = stream, .gpa = t.allocator, .held = &h, .prof = p };
    try t.expect(!try r.open());
    try t.expectEqual(@as(usize, 2), p.n);
    try t.expect(!try r.open());
    try t.expectEqual(@as(usize, 3), p.n);
    try t.expectEqualSlices(prof_mod.Section, &.{ .kda, .gather_a, .head }, p.ends[0..p.n]);
    try t.expectEqual(@as(usize, 3), Ev.external);
    try t.expectEqual(@as(usize, 2), Ev.plain);
    try t.expectError(error.GraphSegments, r.open()); // a third span was never recorded
    p.open = false; // end() would synchronize fake events
}
