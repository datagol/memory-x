/** The memory-x mark: four strands of facts converge on one amber hub and form an x. Each strand is a row of
 *  dots, small at the edge and growing toward the hub, because memx keeps memory as small facts and connects them
 *  through what they share. The amber is the same color the dashboard uses for entities.
 *  Below 28 px a simpler mark with two dots per strand is used, so it stays legible as a favicon. */
const FULL: [number, number, number][] = [
  [8.5, 7.5, 2.4], [14, 11.5, 2.9], [19, 16.6, 3.3],
];
const SMALL: [number, number, number][] = [
  [9.5, 9.2, 3.9], [16.4, 16.4, 4.3],
];

function mirrored(dots: [number, number, number][]) {
  return dots.flatMap(([x, y, r]) => [[x, y, r], [48 - x, y, r], [x, 48 - y, r], [48 - x, 48 - y, r]] as [number, number, number][]);
}

export function LogoMark({ size = 32, title }: { size?: number; title?: string }) {
  const small = size < 28;
  const dots = mirrored(small ? SMALL : FULL);
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      <g fill="var(--logo-dot)">
        {dots.map(([x, y, r], i) => <circle key={i} cx={x} cy={y} r={r} />)}
      </g>
      <circle cx="24" cy="24" r={small ? 5.8 : 5.4} fill="var(--logo-hub)" />
    </svg>
  );
}

/** Mark plus wordmark: "memory" in ink, "-x" in the accent. */
export function Logo({ tagline }: { tagline?: string }) {
  return (
    <div className="brand">
      <LogoMark title="memory-x" />
      <div>
        <b>memory<span className="x">-x</span></b>
        {tagline && <span>{tagline}</span>}
      </div>
    </div>
  );
}
