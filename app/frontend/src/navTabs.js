// The app's navigation vocabulary, in one place.
//
// Start needs the sub-tab ids to roll up busy state onto its two hero buttons,
// and App needs the same lists for the nav itself. Importing them from App would
// be circular (App renders Start), and copying them is how the ids in App's
// prepare-tab fallback drifted from PREPARE_TABS in the first place.

// Ids are the wire values -- they key persisted sub-tab state and every
// setAppMode call site -- so "prepare" keeps its id while showing as Workshop.
export const MODE_TABS = [
  { id: "prepare", label: "Workshop", tip: "Datasets, training, and models" },
  { id: "play", label: "Play", tip: "Create, bend, merge, and sweep" },
];

export const PREPARE_TABS = [
  { id: "data", label: "Data", tip: "Import images and build datasets" },
  { id: "train", label: "Train", tip: "Start or revisit training runs" },
  { id: "models", label: "Models", tip: "Named library models" },
];

export const PLAY_TABS = [
  { id: "create", label: "Create", tip: "Generate, brush regions, post-process, and upscale" },
  { id: "bend", label: "Bend", tip: "Tweak model layers with bend stacks" },
  { id: "merge", label: "Merge", tip: "Blend two models and compare" },
  { id: "sweep", label: "Sweep", tip: "Grid over sampling settings" },
];

export const PREPARE_TAB_IDS = PREPARE_TABS.map((t) => t.id);
export const PLAY_TAB_IDS = PLAY_TABS.map((t) => t.id);

/** True when any of `ids` is currently running work. */
export const anyBusy = (busyTabs, ids) => ids.some((id) => busyTabs?.[id]);
