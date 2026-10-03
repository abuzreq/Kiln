// Where a model now lives, for a path recorded before it moved.
//
// Models used to be kept in the install's own folders (models/pretrained,
// models/fine_tuned, vendor/xurdif/models) and now live in the workspace.
// Recipes inside images, stars and the remembered pick name a model by its
// path, so a model moved out of those folders keeps its file name and a missing
// path there maps to the workspace model of that name. Mirrors _relocated in
// app/core/model_manager.py, which does the same for the server.

const LEGACY_MODEL = /[\\/](?:models[\\/](?:pretrained|fine_tuned)|vendor[\\/]xurdif[\\/]models)[\\/]([^\\/]+\.pt)$/i;

const baseName = (p) => String(p || "").split(/[\\/]/).pop().toLowerCase();

/** `path`, or the workspace model it moved to. Any other missing path is left
 *  alone: matching it to some other model of the same name would quietly use
 *  the wrong one. */
export function relocateModelPath(path, models) {
  if (!path || !models?.length || models.some((m) => m.path === path)) return path;
  const legacy = LEGACY_MODEL.exec(path);
  if (!legacy) return path;
  const name = legacy[1].toLowerCase();
  const hit = models.find((m) => m.source === "workspace" && baseName(m.path) === name);
  return hit ? hit.path : path;
}
