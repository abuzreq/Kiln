import React from "react";

// One place for the small line glyphs the panels use. All are 24-unit stroke
// icons in the current colour, so they take the accent when a seg button is
// on and dim with a disabled button. Every one is aria-hidden: the button's
// text or aria-label is what a reader gets. A few are other icon sets' shapes,
// credited where they are and in THIRD_PARTY_NOTICES.md.

function Icon({ children, size = 14, className = "" }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
      className={`ico ${className}`.trim()}
    >
      {children}
    </svg>
  );
}

export function EyeIcon({ off, size = 15 }) {
  return (
    <Icon size={size}>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
      {off && <line x1="4" y1="4" x2="20" y2="20" />}
    </Icon>
  );
}

/** Two crossing arrows: Feather's "shuffle" (MIT), for drawing a new seed. */
export function ShuffleIcon(props) {
  return (
    <Icon {...props}>
      <polyline points="16 3 21 3 21 8" />
      <line x1="4" y1="20" x2="21" y2="3" />
      <polyline points="21 16 21 21 16 21" />
      <line x1="15" y1="15" x2="21" y2="21" />
      <line x1="4" y1="4" x2="9" y2="9" />
    </Icon>
  );
}

// Brush and wand are Lucide's "brush" and "wand-sparkles" (lucide.dev, ISC):
// the shapes people already know from other editors.
export function BrushIcon(props) {
  return (
    <Icon {...props}>
      <path d="m11 10 3 3" />
      <path d="M6.5 21A3.5 3.5 0 1 0 3 17.5a2.62 2.62 0 0 1-.708 1.792A1 1 0 0 0 3 21z" />
      <path d="M9.969 17.031 21.378 5.624a1 1 0 0 0-3.002-3.002L6.967 14.031" />
    </Icon>
  );
}

export function WandIcon(props) {
  return (
    <Icon {...props}>
      <path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72" />
      <path d="m14 7 3 3" />
      <path d="M5 6v4" />
      <path d="M19 14v4" />
      <path d="M10 2v2" />
      <path d="M7 8H3" />
      <path d="M21 16h-4" />
      <path d="M11 3H9" />
    </Icon>
  );
}

/** Lucide's "folder-open": opens a folder in the file manager. */
export function FolderIcon(props) {
  return (
    <Icon {...props}>
      <path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2" />
    </Icon>
  );
}

export function ShapeIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="12" height="12" rx="1.5" />
      <circle cx="15.5" cy="15.5" r="5.5" />
    </Icon>
  );
}

export function ContrastIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function MoveIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 2v20M2 12h20" />
      <path d="M9 5l3-3 3 3M9 19l3 3 3-3M5 9l-3 3 3 3M19 9l3 3-3 3" />
    </Icon>
  );
}

export function PaintIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v8M8 12h8" />
    </Icon>
  );
}

export function EraseIcon(props) {
  return (
    <Icon {...props}>
      <path d="M4 15l9-9 6 6-9 9H8l-4-4z" />
      <path d="M9 20h11" />
      <path d="M10 9l6 6" />
    </Icon>
  );
}

export function RectIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="5" width="18" height="14" rx="1.5" />
    </Icon>
  );
}

export function EllipseIcon(props) {
  return (
    <Icon {...props}>
      <ellipse cx="12" cy="12" rx="9" ry="7" />
    </Icon>
  );
}

export function PolygonIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 3l9 6.5-3.5 10.5h-11L3 9.5 12 3z" />
    </Icon>
  );
}

export function PatternIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="6" cy="6" r="2" /><circle cx="12" cy="6" r="2" /><circle cx="18" cy="6" r="2" />
      <circle cx="6" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="18" cy="12" r="2" />
      <circle cx="6" cy="18" r="2" /><circle cx="12" cy="18" r="2" /><circle cx="18" cy="18" r="2" />
    </Icon>
  );
}

export function UndoIcon(props) {
  return (
    <Icon {...props}>
      <path d="M9 14L4 9l5-5" />
      <path d="M4 9h10a6 6 0 0 1 0 12h-3" />
    </Icon>
  );
}

export function InvertIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M12 3v18" />
      <path d="M3 5a2 2 0 0 1 2-2h7v18H5a2 2 0 0 1-2-2V5z" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function ClearIcon(props) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9 9l6 6M15 9l-6 6" />
    </Icon>
  );
}

export function NewMaskIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="4 3" />
      <path d="M12 8v8M8 12h8" />
    </Icon>
  );
}

export function AddIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 5v14M5 12h14" />
    </Icon>
  );
}

export function RedoIcon(props) {
  return (
    <Icon {...props}>
      <path d="M15 14l5-5-5-5" />
      <path d="M20 9H10a6 6 0 0 0 0 12h3" />
    </Icon>
  );
}

/** A small downward caret: this opens a menu. */
/** Lucide's "chevron-down" (ISC). */
export function ChevronDownIcon({ size = 12, ...props }) {
  return (
    <Icon size={size} {...props}>
      <path d="m6 9 6 6 6-6" />
    </Icon>
  );
}

/** A page with a folded corner: the document, for the canvas menu. */
export function FileIcon(props) {
  return (
    <Icon {...props}>
      <path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z" />
      <path d="M14 3v5h5" />
    </Icon>
  );
}

export function UpIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 19V5M5 12l7-7 7 7" />
    </Icon>
  );
}

export function DownIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 5v14M5 12l7 7 7-7" />
    </Icon>
  );
}

export function TrashIcon(props) {
  return (
    <Icon {...props}>
      <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" />
      <path d="M10 10v6M14 10v6" />
    </Icon>
  );
}

export function DownloadIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 4v12M6 10l6 6 6-6" />
      <path d="M4 20h16" />
    </Icon>
  );
}

/** Lucide's "file-json" (ISC): a page with braces, for importing and
 *  exporting a JSON file. Two arrows read as sorting. */
export function FileJsonIcon(props) {
  return (
    <Icon {...props}>
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
      <path d="M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1" />
      <path d="M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1" />
    </Icon>
  );
}

/** Two corners pulled apart: show this larger. */
export function ExpandIcon(props) {
  return (
    <Icon {...props}>
      <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
    </Icon>
  );
}

/** Four tiles: a gallery of presets. */
export function TilesIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </Icon>
  );
}

/** A push pin: keep this panel in place. */
export function PinIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 17v5" />
      <path d="M9 3h6l-1 6 3 3v2H7v-2l3-3z" />
    </Icon>
  );
}

/** A die showing five: roll something random. */
export function DiceIcon(props) {
  return (
    <Icon {...props}>
      <rect x="3" y="3" width="18" height="18" rx="4" />
      <g fill="currentColor" stroke="none">
        <circle cx="8" cy="8" r="1.6" />
        <circle cx="16" cy="8" r="1.6" />
        <circle cx="12" cy="12" r="1.6" />
        <circle cx="8" cy="16" r="1.6" />
        <circle cx="16" cy="16" r="1.6" />
      </g>
    </Icon>
  );
}

export function CaptureIcon(props) {
  return (
    <Icon {...props}>
      <path d="M4 8h3l2-3h6l2 3h3v11H4z" />
      <circle cx="12" cy="13" r="3.5" />
    </Icon>
  );
}
