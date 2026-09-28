import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { api } from "./api";
import type { Annotation, EpisodeDetail, EpisodeSummary, EventValue, Segment, TimestampMap } from "./types";

const CUT_EVENT = "branch_point";
const CUT_COLOR = "#f3dd76";

function nearestIndex(values: number[], target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  if (low <= 0) return 0;
  if (low >= values.length) return values.length - 1;
  return Math.abs(values[low] - target) < Math.abs(values[low - 1] - target) ? low : low - 1;
}

function formatTime(ns: number, start: number) {
  return `${((ns - start) / 1e9).toFixed(3)} s`;
}

function withCutPoint(annotation: Annotation, startNs: number, endNs: number): Annotation {
  const normalized = annotation.dataset_decision
    ? annotation
    : { ...annotation, dataset_decision: "undecided" as const };
  if (normalized.events[CUT_EVENT]) return normalized;
  const stable = normalized.events.stable_grasp;
  return {
    ...normalized,
    events: {
      ...normalized.events,
      [CUT_EVENT]: {
        timestamp_ns: stable?.timestamp_ns ?? startNs + Math.round((endNs - startNs) / 2),
        source: "automatic",
        confidence: stable?.confidence ?? 0,
        evidence: { seed_event: stable ? "stable_grasp" : "episode_midpoint" },
        detector_version: stable?.detector_version
      }
    }
  };
}

function TriState({
  label,
  value,
  onChange
}: {
  label: string;
  value: boolean | null | undefined;
  onChange: (value: boolean | null) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value === true ? "yes" : value === false ? "no" : "unknown"} onChange={(event) => onChange(event.target.value === "yes" ? true : event.target.value === "no" ? false : null)}>
        <option value="unknown">Unknown</option>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </label>
  );
}

function Timeline({
  detail,
  annotation,
  playhead,
  segment,
  onSeek,
  onEvent
}: {
  detail: EpisodeDetail;
  annotation: Annotation;
  playhead: number;
  segment?: Segment;
  onSeek: (value: number) => void;
  onEvent: (name: string, value: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const duration = detail.end_ns - detail.start_ns;
  const valueFromPointer = useCallback(
    (clientX: number) => {
      const bounds = ref.current!.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (clientX - bounds.left) / bounds.width));
      return Math.round(detail.start_ns + ratio * duration);
    },
    [detail.start_ns, duration]
  );
  useEffect(() => {
    if (!dragging) return;
    const move = (event: PointerEvent) => onEvent(dragging, valueFromPointer(event.clientX));
    const up = () => setDragging(null);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging, onEvent, valueFromPointer]);
  return (
    <div
      ref={ref}
      className="timeline"
      onPointerDown={(event) => {
        if (event.target === ref.current) onSeek(valueFromPointer(event.clientX));
      }}
      aria-label="Shared episode timeline"
    >
      {segment && (
        <div
          className="segment-band"
          style={{
            left: `${((segment.start_ns - detail.start_ns) / duration) * 100}%`,
            width: `${((segment.end_ns - segment.start_ns) / duration) * 100}%`
          }}
        />
      )}
      <div className="playhead" style={{ left: `${((playhead - detail.start_ns) / duration) * 100}%` }} />
      {annotation.events[CUT_EVENT] && (
        <button
          className="cut-marker"
          style={{
            left: `${((annotation.events[CUT_EVENT].timestamp_ns - detail.start_ns) / duration) * 100}%`,
            background: CUT_COLOR
          }}
          title={`Cut point ${formatTime(annotation.events[CUT_EVENT].timestamp_ns, detail.start_ns)}`}
          aria-label="Episode cut point"
          onPointerDown={(event) => {
            event.stopPropagation();
            setDragging(CUT_EVENT);
            onSeek(annotation.events[CUT_EVENT].timestamp_ns);
          }}
        />
      )}
    </div>
  );
}

function VideoPanel({
  name,
  src,
  videoRef,
  frameIndex
}: {
  name: string;
  src: string;
  videoRef: (node: HTMLVideoElement | null) => void;
  frameIndex: number;
}) {
  return (
    <section className="video-card">
      <header>
        <strong>{name}</strong>
        <span>frame {frameIndex >= 0 ? frameIndex : "—"}</span>
      </header>
      <video ref={videoRef} src={src} muted playsInline preload="metadata" />
    </section>
  );
}

const SignalCharts = memo(function SignalCharts({
  signals,
  detail,
  annotation,
  onSeek
}: {
  signals: Record<string, number[]>;
  detail: EpisodeDetail;
  annotation: Annotation;
  onSeek: (value: number) => void;
}) {
  const cut = annotation.events[CUT_EVENT];
  const eventLines = cut
    ? [{ xAxis: (cut.timestamp_ns - detail.start_ns) / 1e9, lineStyle: { color: CUT_COLOR, opacity: 0.85 } }]
    : [];
  const chart = (
    timestampKey: string,
    lines: Array<[string, string]>,
    title: string,
    height = 220
  ) => {
    const timestamps = signals[timestampKey] || [];
    const series = lines
      .filter(([key]) => signals[key]?.length)
      .map(([key, label], index) => ({
        name: label,
        type: "line",
        showSymbol: false,
        sampling: "lttb",
        data: signals[key].map((value, item) => [(timestamps[item] - detail.start_ns) / 1e9, value]),
        lineStyle: { width: index < 2 ? 1.8 : 1.1 },
        markLine: index === 0 ? { silent: true, symbol: "none", data: eventLines, label: { show: false } } : undefined
      }));
    return (
      <ReactECharts
        style={{ height }}
        option={{
          animation: false,
          backgroundColor: "transparent",
          title: { text: title, textStyle: { color: "#c9d8d0", fontSize: 12, fontWeight: 600 }, left: 12, top: 4 },
          legend: { data: lines.map(([, label]) => label), textStyle: { color: "#95aaa0", fontSize: 10 }, top: 4, right: 16 },
          grid: { left: 52, right: 18, top: 42, bottom: 48 },
          xAxis: { type: "value", min: 0, max: detail.duration_seconds, axisLabel: { color: "#748b80" }, splitLine: { lineStyle: { color: "#25342d" } } },
          yAxis: { type: "value", scale: true, axisLabel: { color: "#748b80" }, splitLine: { lineStyle: { color: "#25342d" } } },
          dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 8 }],
          tooltip: { trigger: "axis" },
          series
        }}
        onEvents={{
          click: (params: any) => {
            if (Array.isArray(params.value)) onSeek(detail.start_ns + Math.round(params.value[0] * 1e9));
          }
        }}
      />
    );
  };
  return (
    <div className="charts">
      {chart(
        "robot_timestamp_ns",
        [
          ["gripper_position", "gripper measured"],
          ["gripper_target", "gripper target"],
          ["joint_velocity_norm", "joint velocity norm"],
          ["eef_linear_velocity", "EEF linear velocity"],
          ["eef_angular_velocity", "EEF angular velocity"]
        ],
        "Robot & gripper"
      )}
      {chart(
        "ati_timestamp_ns",
        [["ati_wrench_2", "Fz"]],
        "ATI Nano17 · Fz"
      )}
      {chart(
        "gelsight_timestamp_ns",
        [
          ["gelsight_image_delta", "image delta"],
          ["gelsight_contact_score", "contact score"]
        ],
        "GelSight derived contact",
        190
      )}
    </div>
  );
});

function Dashboard({ onOpen }: { onOpen: (episode: string) => void }) {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [cameras, setCameras] = useState<Array<Record<string, any>>>([]);
  useEffect(() => {
    Promise.all([api.dashboard(), api.cameraSessions()]).then(([dashboard, cameraRows]) => {
      setData(dashboard);
      setCameras(cameraRows);
    });
  }, []);
  if (!data) return <div className="empty">Loading dashboard…</div>;
  const cards = [
    ["Episodes", data.totals.episodes],
    ["Unreviewed", data.review_status.unreviewed || 0],
    ["Auto proposed", data.review_status.auto_proposed || 0],
    ["Accepted", data.review_status.accepted || 0],
    ["Rejected", data.review_status.rejected || 0],
    ["Recovery", data.review_status.recovery || 0],
    ["Verified", data.review_status.verified || 0],
    ["Missing sensors", data.totals.missing_sensor_episodes],
    ["Average hold", `${data.totals.average_hold_seconds.toFixed(2)} s`],
    ["Segments", data.totals.segments]
  ];
  return (
    <main className="dashboard">
      <div className="metric-grid">
        {cards.map(([label, value]) => (
          <article className="metric" key={label as string}>
            <span>{label}</span>
            <strong>{value}</strong>
          </article>
        ))}
      </div>
      <div className="dashboard-grid">
        <section className="panel">
          <h2>Action branches</h2>
          {Object.entries(data.action_branches).map(([key, value]) => <div className="bar-row" key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}
        </section>
        <section className="panel">
          <h2>Splits · active segments</h2>
          {Object.entries(data.splits).map(([key, value]) => <div className="bar-row" key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}
        </section>
        <section className="panel wide">
          <h2>Camera sessions</h2>
          <div className="camera-sessions">
            {cameras.map((camera) => (
              <article key={camera.camera_session_id}>
                <img src={camera.reference_frame_url} alt="" />
                <div>
                  <strong>{camera.camera_session_id}</strong>
                  <p>{camera.episode_count} episodes · {camera.consistency_class}</p>
                  <button onClick={() => onOpen(camera.reference_episode_id)}>Open reference</button>
                </div>
              </article>
            ))}
          </div>
        </section>
        {Object.entries(data.matrices).map(([name, rows]) => (
          <section className="panel" key={name}>
            <h2>{name.replaceAll("_", " × ")}</h2>
            {(rows as Array<Record<string, any>>).map((row, index) => (
              <div className="bar-row" key={index}><span>{row.swatch_uid} · {row.column}</span><strong>{row.count}</strong></div>
            ))}
          </section>
        ))}
      </div>
    </main>
  );
}

function Review({
  selected,
  list,
  reviewer,
  onSelect,
  onListRefresh
}: {
  selected: string;
  list: EpisodeSummary[];
  reviewer: string;
  onSelect: (id: string) => void;
  onListRefresh: () => void;
}) {
  const [detail, setDetail] = useState<EpisodeDetail | null>(null);
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [version, setVersion] = useState(0);
  const [signals, setSignals] = useState<Record<string, number[]>>({});
  const [timestampMaps, setTimestampMaps] = useState<Record<string, TimestampMap>>({});
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(2);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [selectedSegment, setSelectedSegment] = useState<string>("");
  const [saveState, setSaveState] = useState("Saved");
  const [error, setError] = useState("");
  const [undoTargets, setUndoTargets] = useState<number[]>([]);
  const [redoTargets, setRedoTargets] = useState<number[]>([]);
  const videoNodes = useRef<Record<string, HTMLVideoElement | null>>({});
  const dirty = useRef(false);
  const loading = useRef(false);
  const saving = useRef(false);

  const load = useCallback(async () => {
    loading.current = true;
    setError("");
    try {
      const [episode, signalData, external, wrist, gelsight, history] = await Promise.all([
        api.episode(selected),
        api.signals(selected),
        api.timestamps(selected, "external"),
        api.timestamps(selected, "wrist"),
        api.timestamps(selected, "gelsight"),
        api.history(selected)
      ]);
      setDetail(episode);
      setAnnotation(withCutPoint(episode.annotation.annotation, episode.start_ns, episode.end_ns));
      setVersion(episode.annotation.version);
      setSignals(signalData);
      setTimestampMaps({ external, wrist, gelsight });
      setPlayhead(episode.start_ns);
      setSegments(episode.segments);
      setSelectedSegment("");
      setUndoTargets(history.slice(1).map((item) => item.version));
      setRedoTargets([]);
      dirty.current = false;
      setSaveState("Saved");
    } catch (reason) {
      setError(String(reason));
    } finally {
      loading.current = false;
    }
  }, [selected]);
  useEffect(() => { load(); }, [load]);

  const change = useCallback((recipe: (current: Annotation) => Annotation) => {
    setAnnotation((current) => {
      if (!current) return current;
      dirty.current = true;
      setSaveState("Unsaved");
      return recipe(current);
    });
  }, []);

  const save = useCallback(async (reason = "autosave") => {
    if (!annotation || !dirty.current || loading.current || saving.current) return;
    saving.current = true;
    setSaveState("Saving…");
    try {
      const oldVersion = version;
      const result = await api.save(selected, annotation, reviewer, version, reason);
      setVersion(result.version);
      setSegments(result.segments);
      setUndoTargets((values) => [oldVersion, ...values.filter((item) => item !== oldVersion)]);
      setRedoTargets([]);
      dirty.current = false;
      setSaveState(`Saved v${result.version}`);
      onListRefresh();
    } catch (reason) {
      setSaveState("Save failed");
      setError(String(reason));
    } finally {
      saving.current = false;
    }
  }, [annotation, onListRefresh, reviewer, selected, version]);
  useEffect(() => {
    if (!dirty.current) return;
    const timer = window.setTimeout(() => save(), 900);
    return () => window.clearTimeout(timer);
  }, [annotation, save]);

  useEffect(() => {
    if (!annotation || !detail) return;
    const timer = window.setTimeout(() => {
      api.preview(selected, annotation).then(setSegments).catch(() => undefined);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [annotation, detail, selected]);

  const seek = useCallback((value: number) => {
    if (!detail) return;
    setPlayhead(Math.max(detail.start_ns, Math.min(detail.end_ns, value)));
  }, [detail]);

  useEffect(() => {
    for (const [stream, map] of Object.entries(timestampMaps)) {
      const node = videoNodes.current[stream];
      if (!node || !map.timestamp_ns.length) continue;
      const index = nearestIndex(map.timestamp_ns, playhead);
      const target = map.media_seconds[index];
      const tolerance = playing ? 0.45 : 0.045;
      if (Number.isFinite(target) && Math.abs(node.currentTime - target) > tolerance) node.currentTime = target;
    }
  }, [playhead, playing, timestampMaps]);

  useEffect(() => {
    for (const node of Object.values(videoNodes.current)) {
      if (!node) continue;
      node.playbackRate = rate;
      if (playing) node.play().catch(() => undefined);
      else node.pause();
    }
  }, [playing, rate]);

  useEffect(() => {
    if (!playing || !detail) return;
    let frame = 0;
    let last = performance.now();
    let lastUiUpdate = 0;
    const loop = (now: number) => {
      if (now - lastUiUpdate < 50) {
        frame = requestAnimationFrame(loop);
        return;
      }
      const delta = (now - last) / 1000;
      last = now;
      lastUiUpdate = now;
      setPlayhead((current) => {
        const selectedItem = segments.find((item) => item.segment_id === selectedSegment);
        const end = selectedItem?.end_ns ?? detail.end_ns;
        const next = current + delta * rate * 1e9;
        if (next >= end) {
          setPlaying(false);
          return selectedItem?.start_ns ?? detail.start_ns;
        }
        return next;
      });
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [detail, playing, rate, segments, selectedSegment]);

  const frameIndex = (stream: string) => {
    const map = timestampMaps[stream];
    return map ? nearestIndex(map.timestamp_ns, playhead) : -1;
  };
  const robotTimes = signals.robot_timestamp_ns || [];
  const stepFrame = useCallback((direction: number) => {
    if (!robotTimes.length) return;
    const index = nearestIndex(robotTimes, playhead);
    seek(robotTimes[Math.max(0, Math.min(robotTimes.length - 1, index + direction))]);
  }, [playhead, robotTimes, seek]);
  const jumpToCut = useCallback(() => {
    if (!annotation) return;
    const cut = annotation.events[CUT_EVENT];
    if (cut) seek(cut.timestamp_ns);
  }, [annotation, seek]);

  const commitImmediate = useCallback(async (patch: Partial<Annotation>, reason: string) => {
    if (!annotation || loading.current || saving.current) return;
    const next = { ...annotation, ...patch } as Annotation;
    setAnnotation(next);
    dirty.current = false;
    saving.current = true;
    setSaveState("Saving…");
    setError("");
    try {
      const oldVersion = version;
      const result = await api.save(selected, next, reviewer, version, reason);
      setVersion(result.version);
      setSegments(result.segments);
      setUndoTargets((values) => [oldVersion, ...values.filter((item) => item !== oldVersion)]);
      setRedoTargets([]);
      setSaveState(`Saved v${result.version}`);
      onListRefresh();
    } catch (reason) {
      dirty.current = true;
      setSaveState("Save failed");
      setError(String(reason));
    } finally {
      saving.current = false;
    }
  }, [annotation, onListRefresh, reviewer, selected, version]);

  const commitBranch = useCallback((branch: "remove" | "leave") => {
    void commitImmediate({ action_branch: branch }, `manual branch label: ${branch}`);
  }, [commitImmediate]);

  const commitDecision = useCallback((decision: "keep" | "drop") => {
    void commitImmediate({ dataset_decision: decision }, `manual dataset decision: ${decision}`);
  }, [commitImmediate]);

  const setReview = useCallback((status: Annotation["review_status"]) => {
    if (!annotation) return;
    if (status === "accepted" && annotation.action_branch === "unknown") {
      setError("Choose Remove or Leave before accepting.");
      return;
    }
    change((current) => ({ ...current, review_status: status }));
  }, [annotation, change]);
  const navigate = useCallback((direction: number) => {
    const index = list.findIndex((item) => item.episode_id === selected);
    const next = list[index + direction];
    if (next) onSelect(next.episode_id);
  }, [list, onSelect, selected]);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.code === "Space") { event.preventDefault(); setPlaying((value) => !value); }
      else if (event.key === "ArrowLeft") stepFrame(-1);
      else if (event.key === "ArrowRight") stepFrame(1);
      else if (event.key.toLowerCase() === "n") navigate(1);
      else if (event.key.toLowerCase() === "p") navigate(-1);
      else if (event.key.toLowerCase() === "r") commitBranch("remove");
      else if (event.key.toLowerCase() === "l") commitBranch("leave");
      else if (event.key.toLowerCase() === "k") commitDecision("keep");
      else if (event.key.toLowerCase() === "x") commitDecision("drop");
      else if (event.key.toLowerCase() === "u") change((current) => ({ ...current, action_branch: "unknown" }));
      else if (event.key === "1") setReview("accepted");
      else if (event.key === "2") setReview("rejected");
      else if (event.key === "3") setReview("recovery");
    };
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  }, [change, commitBranch, commitDecision, navigate, setReview, stepFrame]);

  const restore = async (operation: "undo" | "redo") => {
    const targets = operation === "undo" ? undoTargets : redoTargets;
    if (!targets.length) return;
    const target = targets[0];
    try {
      const result = await api.restore(selected, operation, target, reviewer);
      if (operation === "undo") {
        setUndoTargets((items) => items.slice(1));
        setRedoTargets((items) => [version, ...items]);
      } else {
        setRedoTargets((items) => items.slice(1));
        setUndoTargets((items) => [version, ...items]);
      }
      setAnnotation(detail ? withCutPoint(result.annotation, detail.start_ns, detail.end_ns) : result.annotation);
      setVersion(result.version);
      setSegments(result.segments);
      dirty.current = false;
      setSaveState(`Saved v${result.version}`);
    } catch (reason) {
      setError(String(reason));
    }
  };

  if (!detail || !annotation) return <div className="empty">Loading episode… {error}</div>;
  const activeSegment = segments.find((item) => item.segment_id === selectedSegment);
  const currentRobotFrame = robotTimes.length ? nearestIndex(robotTimes, playhead) : -1;
  return (
    <div className="review-layout">
      <aside className="episode-rail">
        <header><strong>{list.length} episodes</strong><span>Pass 1 queue</span></header>
        {list.map((item) => (
          <button key={item.episode_id} className={item.episode_id === selected ? "active" : ""} onClick={() => onSelect(item.episode_id)}>
            <i className={`status-dot ${item.severity}`} />
            <span><strong>{item.episode_id.replace("episode_", "")}</strong><small>{item.review_status} · {item.action_branch}</small></span>
            <em>{item.quality_score?.toFixed(2) ?? "—"}</em>
          </button>
        ))}
      </aside>
      <main className="review-main">
        <div className="episode-heading">
          <div>
            <p>{detail.session_id} · {detail.swatch_uid} · {detail.split} · Auto-trimmed {(detail.trimmed_leading_seconds || 0).toFixed(2)} s stationary frames{detail.camera_streams_swapped ? " · Camera names corrected" : ""}</p>
            <h1>{detail.episode_id}</h1>
          </div>
          <div className="save-state"><span className={`status-dot ${detail.qc?.severity || "unknown"}`} />{saveState}</div>
        </div>
        {error && <div className="error-banner" onClick={() => setError("")}>{error}</div>}
        <div className="video-grid">
          <VideoPanel name="D435 external RGB" src={detail.streams.external.media_url!} videoRef={(node) => { videoNodes.current.external = node; }} frameIndex={frameIndex("external")} />
          <VideoPanel name="Wrist / fisheye RGB" src={detail.streams.wrist.media_url!} videoRef={(node) => { videoNodes.current.wrist = node; }} frameIndex={frameIndex("wrist")} />
          <VideoPanel name="GelSight" src={detail.streams.gelsight.media_url!} videoRef={(node) => { videoNodes.current.gelsight = node; }} frameIndex={frameIndex("gelsight")} />
        </div>
        <div className="transport">
          <button onClick={jumpToCut}>Cut point</button>
          <button onClick={() => stepFrame(-1)}>‹ Frame</button>
          <button className="play" onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
          <button onClick={() => stepFrame(1)}>Frame ›</button>
          {[1, 2, 5, 10].map((value) => <button className={rate === value ? "selected" : ""} key={value} onClick={() => setRate(value)}>{value}×</button>)}
          <span className="timestamp">t={formatTime(playhead, detail.start_ns)} · raw {Math.round(playhead)} ns · policy frame {currentRobotFrame}</span>
        </div>
        <Timeline
          detail={detail}
          annotation={annotation}
          playhead={playhead}
          segment={activeSegment}
          onSeek={seek}
          onEvent={(name, timestamp) => {
            change((current) => ({
              ...current,
              events: { ...current.events, [name]: { ...current.events[name], timestamp_ns: timestamp, source: "manual", confidence: 1 } }
            }));
          }}
        />
        <div className="workspace-grid">
          <div>
            <SignalCharts signals={signals} detail={detail} annotation={annotation} onSeek={seek} />
          </div>
          <aside className="inspector">
            <section>
              <div className="section-title"><h2>Keep episode</h2><span>K / X · Click to save</span></div>
              <div className="branch-buttons decision-buttons">
                <button
                  className={annotation.dataset_decision === "keep" ? "active keep" : "keep"}
                  aria-pressed={annotation.dataset_decision === "keep"}
                  disabled={saveState === "Saving…"}
                  onClick={() => commitDecision("keep")}
                ><strong>Keep this episode</strong><span>KEEP · K</span></button>
                <button
                  className={annotation.dataset_decision === "drop" ? "active drop" : "drop"}
                  aria-pressed={annotation.dataset_decision === "drop"}
                  disabled={saveState === "Saving…"}
                  onClick={() => commitDecision("drop")}
                ><strong>Drop this episode</strong><span>DROP · X</span></button>
              </div>
            </section>
            <section>
              <div className="section-title"><h2>Action branch</h2><span>R / L · Click to save</span></div>
              <div className="branch-buttons">
                <button
                  className={annotation.action_branch === "remove" || annotation.action_branch.startsWith("legacy_") ? "active remove" : "remove"}
                  aria-pressed={annotation.action_branch === "remove" || annotation.action_branch.startsWith("legacy_")}
                  disabled={saveState === "Saving…"}
                  onClick={() => commitBranch("remove")}
                ><strong>Pick and remove</strong><span>REMOVE · R</span></button>
                <button
                  className={annotation.action_branch === "leave" ? "active leave" : "leave"}
                  aria-pressed={annotation.action_branch === "leave"}
                  disabled={saveState === "Saving…"}
                  onClick={() => commitBranch("leave")}
                ><strong>Pick and leave</strong><span>LEAVE · L</span></button>
              </div>
            </section>
            <section>
              <div className="section-title"><h2>Pass 1 review</h2><span>1 / 2 / 3</span></div>
              <div className="button-row">
                <button className="accept" onClick={() => setReview("accepted")}>Accept</button>
                <button className="reject" onClick={() => setReview("rejected")}>Reject</button>
                <button onClick={() => setReview("recovery")}>Recovery</button>
                <button onClick={() => setReview("verified")}>Verified</button>
              </div>
              <label className="field"><span>Swatch UID</span><input value={annotation.swatch_uid || ""} onChange={(event) => change((current) => ({ ...current, swatch_uid: event.target.value }))} /></label>
              <TriState label="Single layer" value={annotation.single_layer} onChange={(value) => change((current) => ({ ...current, single_layer: value }))} />
              <TriState label="Correct grasp region" value={annotation.correct_grasp_region} onChange={(value) => change((current) => ({ ...current, correct_grasp_region: value }))} />
              <TriState label="Stable hold present" value={annotation.stable_hold_present} onChange={(value) => change((current) => ({ ...current, stable_hold_present: value }))} />
              <TriState label="Sensor complete" value={annotation.sensor_complete} onChange={(value) => change((current) => ({ ...current, sensor_complete: value }))} />
              <label className="field"><span>Failure reason</span><input value={annotation.failure_reason} onChange={(event) => change((current) => ({ ...current, failure_reason: event.target.value }))} /></label>
            </section>
            <section>
              <div className="section-title"><h2>Single cut point</h2><span>Drag the yellow marker</span></div>
              <button className="cut-summary" onClick={jumpToCut}>
                <strong>{formatTime(annotation.events[CUT_EVENT].timestamp_ns, detail.start_ns)}</strong>
                <span>First half: grasp check · Second half: branch action</span>
              </button>
              <div className="button-row">
                <button disabled={!undoTargets.length} onClick={() => restore("undo")}>Undo</button>
                <button disabled={!redoTargets.length} onClick={() => restore("redo")}>Redo</button>
                <button onClick={() => save("manual save")}>Save now</button>
              </div>
            </section>
            <section>
              <div className="section-title"><h2>Segment preview</h2><span>{segments.length} proposed</span></div>
              <div className="segment-list">
                {segments.map((segment) => (
                  <button key={segment.segment_id} className={selectedSegment === segment.segment_id ? "active" : ""} onClick={() => { setSelectedSegment(segment.segment_id); seek(segment.start_ns); setPlaying(false); }}>
                    <strong>{segment.segment_type}</strong><span>{segment.duration.toFixed(2)} s</span><small>{segment.prompt}</small>
                    {!!segment.warnings.length && <em>{segment.warnings.join(", ")}</em>}
                  </button>
                ))}
              </div>
              {activeSegment && <p className="segment-range">start {formatTime(activeSegment.start_ns, detail.start_ns)} → end {formatTime(activeSegment.end_ns, detail.start_ns)}</p>}
            </section>
            <section>
              <div className="section-title"><h2>QC</h2><span>{detail.qc?.severity}</span></div>
              {detail.qc?.hard_failures.map((item) => <p className="qc red" key={item}>{item}</p>)}
              {detail.qc?.warnings.map((item) => <p className="qc yellow" key={item}>{item}</p>)}
              {Object.entries(detail.qc?.scores || {}).map(([key, value]) => <div className="score" key={key}><span>{key.replaceAll("_", " ")}</span><progress value={value} max={1} /><strong>{value.toFixed(2)}</strong></div>)}
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<"dashboard" | "review">("review");
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [selected, setSelected] = useState("");
  const [reviewer, setReviewer] = useState(localStorage.getItem("curator-reviewer") || "suhang");
  const [filter, setFilter] = useState("");
  const refresh = useCallback(() => {
    api.episodes(filter).then((result) => {
      setEpisodes(result.items);
      if (!selected && result.items.length) setSelected(result.items[0].episode_id);
    });
  }, [filter, selected]);
  useEffect(() => { refresh(); }, [refresh]);
  const open = (episode: string) => { setSelected(episode); setTab("review"); };
  return (
    <div className="app">
      <nav className="topbar">
        <div className="brand"><span>FD</span><strong>Fabric-DROID</strong><em>Dataset Curator</em></div>
        <div className="tabs"><button className={tab === "dashboard" ? "active" : ""} onClick={() => setTab("dashboard")}>Dashboard</button><button className={tab === "review" ? "active" : ""} onClick={() => setTab("review")}>Episode review</button></div>
        <div className="nav-actions">
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="">All episodes</option><option value="review_status=unreviewed">Unreviewed</option><option value="review_status=accepted">Accepted</option><option value="severity=red">Red QC</option><option value="severity=yellow">Yellow QC</option><option value="severity=green">Green QC</option>
          </select>
          <label>Reviewer <input value={reviewer} onChange={(event) => { setReviewer(event.target.value); localStorage.setItem("curator-reviewer", event.target.value); }} /></label>
        </div>
      </nav>
      {tab === "dashboard" ? <Dashboard onOpen={open} /> : selected ? <Review selected={selected} list={episodes} reviewer={reviewer} onSelect={setSelected} onListRefresh={refresh} /> : <div className="empty">Run the indexer to add episodes.</div>}
    </div>
  );
}
