import type { Annotation, EpisodeDetail, EpisodeSummary, Segment, TimestampMap } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${body}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  episodes: (params = "") =>
    request<{ total: number; items: EpisodeSummary[] }>(`/api/episodes${params ? `?${params}` : ""}`),
  episode: (id: string) => request<EpisodeDetail>(`/api/episodes/${id}`),
  signals: (id: string) => request<Record<string, number[]>>(`/api/episodes/${id}/signals`),
  timestamps: (id: string, stream: string) =>
    request<TimestampMap>(`/api/episodes/${id}/timestamps/${stream}`),
  dashboard: () => request<Record<string, any>>("/api/dashboard"),
  cameraSessions: () => request<Array<Record<string, any>>>("/api/camera-sessions"),
  preview: (id: string, annotation: Annotation) =>
    request<Segment[]>(`/api/episodes/${id}/segments/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotation)
    }),
  save: (id: string, annotation: Annotation, reviewer: string, expectedVersion: number, reason = "autosave") =>
    request<{ version: number; annotation: Annotation; segments: Segment[] }>(`/api/episodes/${id}/annotation`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        annotation,
        reviewer,
        reason,
        expected_version: expectedVersion
      })
    }),
  history: (id: string) =>
    request<Array<{ version: number; annotation: Annotation; reviewer: string; reason: string }>>(
      `/api/episodes/${id}/annotations/history`
    ),
  restore: (id: string, operation: "undo" | "redo" | "restore", version: number, reviewer: string) =>
    request<{ version: number; annotation: Annotation; segments: Segment[] }>(
      `/api/episodes/${id}/annotations/${operation}/${version}?reviewer=${encodeURIComponent(reviewer)}`,
      { method: "POST" }
    )
};
