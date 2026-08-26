import React from "react";
import { Select } from "./ui.jsx";
import { THUMB_LEVELS } from "./LazySourceGallery.jsx";

/** The app's one sort vocabulary. Three screens used to declare their own,
 *  and disagreed on whether "date" was labelled "Date" or "Date added". */
export const GALLERY_SORT_OPTS = [
  { value: "name", label: "Name" },
  { value: "date", label: "Date added" },
];

/** The S/M/L thumbnail-size control, shared by the galleries and the Play model picker. */
export function ThumbSizeSeg({ value, onChange, levels = THUMB_LEVELS, ariaLabel = "Thumbnail size" }) {
  return (
    <div className="seg seg-sm" role="group" aria-label={ariaLabel}>
      {levels.map((t) => (
        <button
          key={t.id}
          type="button"
          className={value === t.id ? "on" : ""}
          aria-pressed={value === t.id}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function ThumbGalleryToolbar({
  title,
  count,
  sort,
  onSortChange,
  thumbLevel,
  onThumbLevelChange,
  sortOptions = GALLERY_SORT_OPTS,
  levels = THUMB_LEVELS,
  heading = true,
}) {
  return (
    <div className="row between center wrap gap-2 mb-2">
      {heading
        ? <h3 className="mb-0">{title} ({count})</h3>
        : <span className="sub">{title} ({count})</span>}
      <div className="row gap-1 wrap center">
        <Select label="Sort" value={sort} onChange={onSortChange} options={sortOptions} tip="Sort order" />
        <ThumbSizeSeg levels={levels} value={thumbLevel} onChange={onThumbLevelChange} />
      </div>
    </div>
  );
}
