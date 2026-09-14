import React from "react";

// One place for the small line glyphs the panels use. All are 24-unit stroke
// icons in the current colour, so they take the accent when a seg button is
// on and dim with a disabled button. Every one is aria-hidden: the button's
// text or aria-label is what a reader gets.

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

/** Two crossing arrows: the shuffle glyph, for drawing a new seed. */
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

export function BrushIcon(props) {
  return (
    <Icon {...props}>
      <path d="M14 3l7 7-8.5 8.5a3 3 0 0 1-4.2 0L7 17.2l7-7" />
      <path d="M7 17.2L3 21" />
      <path d="M10.5 6.5l7 7" />
    </Icon>
  );
}

export function WandIcon(props) {
  return (
    <Icon {...props}>
      <path d="M15 4l5 5L7 22l-5-5L15 4z" />
      <path d="M12 7l5 5" />
      <path d="M19 2v3M17.5 3.5h3M4 10v2M3 11h2" />
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

export function GenerateIcon(props) {
  return (
    <Icon {...props}>
      <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15z" />
      <path d="M5 16l.6 1.4L7 18l-1.4.6L5 20l-.6-1.4L3 18l1.4-.6L5 16z" />
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
