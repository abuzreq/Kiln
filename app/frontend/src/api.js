// Thin API client for the Kiln backend. Same-origin in production; proxied in dev.

async function handle(res) {
  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  if (!res.ok || body.ok === false) {
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return body.data !== undefined ? body.data : body;
}

export const api = {
  get: (path, opts) => fetch(`/api${path}`, opts).then(handle),
  post: (path, data, opts) =>
    fetch(`/api${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data || {}),
      ...opts,
    }).then(handle),
  put: (path, data, opts) =>
    fetch(`/api${path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data || {}),
      ...opts,
    }).then(handle),
  del: (path, data) =>
    fetch(`/api${path}`, {
      method: "DELETE",
      headers: data ? { "Content-Type": "application/json" } : undefined,
      body: data ? JSON.stringify(data) : undefined,
    }).then(handle),
  upload: (path, formData) =>
    fetch(`/api${path}`, { method: "POST", body: formData }).then(handle),
};

/** The model list, streamed: `onProgress` gets the list so far each time a
 *  model arrives, and the promise resolves with the whole list. The server
 *  writes one JSON object per line and ends with {"done": true}; a stream that
 *  stops before that is an error, so a dropped connection never passes for a
 *  short list. */
export async function streamModels(onProgress) {
  const res = await fetch("/api/models?stream=1");
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const list = [];
  let buf = "";
  let done = false;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    buf += value;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.error) throw new Error(msg.error);
      if (msg.done) done = true;
      else if (msg.model) {
        list.push(msg.model);
        onProgress?.([...list]);
      }
    }
  }
  if (!done) throw new Error("The model list stopped before it finished");
  return list;
}

/** POST and save the response as a file. Used for exports that must keep their
 *  embedded metadata — a data-URL <a download> would strip it. */
export async function downloadPost(path, data, filename) {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data || {}),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).error || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  // the body went straight to disk, so headers are the only way back for
  // anything the server wants to tell the caller about the export
  return res.headers;
}

// Media helpers (return URLs the <img> tag can use directly).
export const mediaUrl = (path) => `/api/media?path=${encodeURIComponent(path)}`;
export const thumbUrl = (path) => `/api/thumb?path=${encodeURIComponent(path)}`;

// Poll a job until it finishes; onUpdate receives each snapshot.
export async function pollJob(jobId, onUpdate, interval = 350) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await api.get(`/jobs/${jobId}`);
        if (!job) return reject(new Error("job disappeared"));
        onUpdate && onUpdate(job);
        if (["done", "error", "cancelled"].includes(job.status)) {
          return resolve(job);
        }
        setTimeout(tick, interval);
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}
