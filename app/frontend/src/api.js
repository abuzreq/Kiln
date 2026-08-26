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
  del: (path, data) =>
    fetch(`/api${path}`, {
      method: "DELETE",
      headers: data ? { "Content-Type": "application/json" } : undefined,
      body: data ? JSON.stringify(data) : undefined,
    }).then(handle),
  upload: (path, formData) =>
    fetch(`/api${path}`, { method: "POST", body: formData }).then(handle),
};

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
