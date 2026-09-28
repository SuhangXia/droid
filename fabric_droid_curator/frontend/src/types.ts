export type EventValue = {
  timestamp_ns: number;
  source: "automatic" | "manual" | "accepted_automatic";
  confidence: number;
  evidence: Record<string, unknown>;
  detector_version?: string | null;
};

export type Annotation = {
  episode_id: string;
  session_id?: string | null;
  swatch_uid?: string | null;
  source_slot?: string | null;
  operator?: string | null;
  camera_session_id?: string | null;
  success?: boolean | null;
  review_status: "unreviewed" | "auto_proposed" | "accepted" | "rejected" | "recovery" | "verified";
  failure_reason: string;
  single_layer?: boolean | null;
  correct_grasp_region?: boolean | null;
  stable_hold_present?: boolean | null;
  sensor_complete?: boolean | null;
  action_branch: "remove" | "leave" | "legacy_green_left" | "legacy_white_right" | "unknown";
  dataset_decision: "keep" | "drop" | "undecided";
  events: Record<string, EventValue>;
  material_labels: Record<string, unknown>;
  notes: string;
};

export type Segment = {
  segment_id: string;
  source_episode_id: string;
  segment_type: string;
  start_ns: number;
  end_ns: number;
  duration: number;
  instruction_template_id: string;
  prompt: string;
  dataset_decision?: "keep" | "drop" | "undecided";
  split_point_ns?: number | null;
  warnings: string[];
};

export type EpisodeSummary = {
  episode_id: string;
  session_id: string;
  duration_seconds: number;
  review_status: string;
  action_branch: string;
  swatch_uid: string;
  split: string;
  camera_session_id: string;
  start_pose_bucket: string;
  annotation_version: number;
  severity: "red" | "yellow" | "green" | "unknown";
  quality_score?: number | null;
  thumbnail_url: string;
};

export type EpisodeDetail = {
  episode_id: string;
  session_id: string;
  raw_path: string;
  duration_seconds: number;
  start_ns: number;
  raw_start_ns?: number;
  trimmed_leading_seconds?: number;
  camera_streams_swapped?: boolean;
  end_ns: number;
  review_status: string;
  action_branch: string;
  swatch_uid: string;
  split: string;
  camera_session_id: string;
  metadata: Record<string, unknown>;
  streams: Record<string, {
    count: number;
    start_ns: number;
    end_ns: number;
    measured_hz: number;
    encoded_fps?: number | null;
    media_url?: string | null;
  }>;
  proposals: Array<{
    event_name: string;
    candidate_timestamp_ns: number;
    confidence: number;
    evidence: Record<string, unknown>;
    detector_version: string;
  }>;
  qc?: {
    severity: "red" | "yellow" | "green";
    scores: Record<string, number>;
    hard_failures: string[];
    warnings: string[];
    metrics: Record<string, unknown>;
  };
  segments: Segment[];
  annotation: {
    version: number;
    annotation: Annotation;
    reviewer?: string;
    reason?: string;
  };
};

export type TimestampMap = {
  timestamp_ns: number[];
  relative_seconds: number[];
  media_seconds: number[];
  encoded_fps: number;
};
