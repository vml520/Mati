/* Built by tools/build/build_avatar.sh — do not edit. */
/* Source: media/axolotl river animation/*.jsx */
;(function(){ /* ── animations-v3.jsx ── */
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
// Copied omelette starter. Re-running copy_starter_component with this kind overwrites this file with the latest version (page content is unaffected).

/* BEGIN USAGE */
// animations-v3.jsx — continuous-composition animation engine.
//
// THE MODEL: the animation is ONE element tree rendered as a pure function
// of one authored-time axis. Nothing mounts or unmounts at section
// boundaries, so any element can move, morph, or persist across them by
// ordinary interpolation. The scene list (OM_SCENES) is the user-control
// view — names, order, playback durations — and the engine derives the cue
// table from it, so structure has exactly one source and cannot drift.
//
// API INDEX (every export is a window global):
//   <CompositionStage width height scenes={window.OM_SCENES}
//                     playback={window.OM_PLAYBACK} bg>
//     <Piece />   — ONE component, the whole animation
//   </CompositionStage>
//   useComposition() -> {T, CUES, time, duration, authoredTotal, playing}
//     T: authored seconds (warped per-section by user trims/speeds) —
//        key ALL choreography to T, never to wall-clock time
//     CUES: {SectionName: authoredStart} derived from OM_SCENES; an unknown
//        name returns NaN and raises a preview-only badge (never exports);
//        duplicate section names bind to the first occurrence
//   <Shot from={CUES.Build} to={CUES.Close}> — children visible between two
//     authored times (an authored hard cut in one line); children stay
//     mounted (media keeps its readiness) and are hidden outside the window
//   <Captions items={[{at, until?, text}, ...]} /> — ONE caption element,
//     at most one visible at a time, keyed to T; 'until' defaults to the
//     next item's 'at'; a last item with no 'until' stays to the end
//   WATERCOLOR (only when the Watercolor illustration skill is active —
//   otherwise ignore these entries). A painting is a function(p) written
//   against the paint kit, on a width x height sheet; it needs
//   watercolor_kit.js loaded by a <script> tag before this engine, must
//   be a stable function defined once (module scope, never an inline
//   arrow), and every component below renders <img> elements, so it all
//   exports by construction.
//   <WatercolorPainting painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed scale quality style /> — the painting assembled from its own
//     STROKES: each wash / ink line / splatter is a separate layer
//     stacked over the paper, appearing in painting order between two
//     authored times (washes bloom in, ink draws tip to tail). This is the
//     default way to show a watercolor being painted. It keeps the sheet's
//     aspect ratio (size it with style, e.g. {position:'absolute', left,
//     top, width}). scale is the layers' render resolution over width x
//     height (default 1, a deliberate weight-over-dpi trade — raise it
//     toward the zoom factor if the composition zooms into the painting,
//     or toward the devicePixelRatio for a hero-sized sheet); quality is
//     0..1 layer image quality (default 0.92; 1 is the encoder's maximum).
//   useWatercolorLayers(fn, {width, height, seed, scale, quality}) -> L
//     (null if the kit isn't loaded — load watercolor_kit.js before the
//     engine — or if the painting fails to build) — the painting taken apart into strokes, for
//     choreography beyond in-order painting: L.count strokes, L.kind(i)
//     ('wash' | 'gradedWash' | 'glaze' | 'ink' | 'hatch' | 'splatter' |
//     'dryStroke' | 'reserve' | 'caption'), L.span(i) = the stroke's
//     {from, to} share of the painting's 0..1 timeline; call L.warm()
//     once after load so finished strokes pre-render off the critical
//     path (WatercolorPainting does this itself). Compose with:
//   <WatercolorSheet layers={L} style>children</WatercolorSheet> — the
//     paper the strokes sit on (keeps the sheet's aspect ratio), and
//   <WatercolorStroke index={i} at={0..1} style /> — stroke i as its own
//     element, placed where it was painted; at is its painting progress
//     (0 hidden, 1 finished — drive it from T with animate()); style lets
//     you move, scale, rotate, or fade the stroke (transform / opacity).
//     Strokes are paint, so they multiply: overlapping strokes darken
//     where they cross, as in the still image, within a few 8-bit levels
//     (tighter still at quality 1). The sheet clips to its
//     box — for strokes that fly in from outside it, set
//     style={{overflow: 'visible'}} on the WatercolorSheet. 'reserve' strokes
//     are erasures (lifted paper) — keep them where they were painted and
//     reveal them in order after the strokes they erase; moving an erase
//     around has no sensible meaning.
//   <WatercolorReveal painting={fn} from={CUES.X} to={CUES.Y} width height
//     seed steps scale format quality style />, or <WatercolorReveal
//     frames={[src, ...]} from to /> — the whole painting as ONE flat
//     image that paints on (frames pre-baked in the background, so it is
//     the lightest option and the one to zoom or pan over as a single
//     picture). Prefer WatercolorPainting when the strokes themselves
//     should appear one by one or be individually animated. format is
//     the image MIME type (default image/jpeg), quality 0..1 (default
//     0.88). Frames bake at width x height times scale (default: the
//     device pixel ratio, capped at 2) — if the composition zooms INTO
//     the painting, raise scale toward the maximum zoom so frames stay
//     crisp. The kit caps a sheet at ~12M pixels and the components clamp
//     scale to stay under it; exported video sharpness also depends on
//     the export dialog's own resolution choice.
//   Motion: Easing.{linear, easeIn|Out|InOutQuad/Cubic/Quart/Expo/Sine,
//     easeIn|Out|InOutBack, easeOutElastic}, interpolate(input, output, ease),
//     animate({from, to, start, end, ease}) -> fn(T), clamp(v, min, max)
//   Plumbing (rarely needed): Stage, PlaybackBar, TimelineContext,
//     useTime, useTimeline
//   Seek event (host/export transport): 'data-om-seek-to-time-frame',
//     detail {time, sync, playing} — the stage owns it; never implement it
//     yourself
//
// THE AUTHORING CONTRACT — this is what makes the host timeline's trim and
// speed gestures write back into YOUR file, so follow it exactly:
//   1. Declare the scene list as a JSON string literal in a plain inline
//      <script> of the main document (NOT type="text/babel", NOT a sibling
//      .jsx — only vanilla inline scripts are addressable for write-back):
//        <script>window.OM_SCENES = '[{"name":"Opening","dur":3,"desc":"The logo fades in and the title settles"},{"name":"Build","dur":5,"desc":"Bars grow to their final values"}]';</script>
//      Give every entry a "desc": one short plain-words sentence saying
//      what happens in that section. The user reads it in the timeline's
//      section popover — keep it true whenever you edit the section.
//   2. Pass the string through untouched:
//        <CompositionStage scenes={window.OM_SCENES} ...>
//   3. ALSO declare the playback setting the same way:
//        <script>window.OM_PLAYBACK = '{"mode":"loop"}';</script>
//      and pass it through untouched (values: '{"mode":"loop"}' or
//      '{"mode":"times","count":N}'; omitting keeps loop behavior but
//      leaves the host Repeat control read-only for this document).
//   IMPORTANT — the exportable-video contract: CompositionStage/Stage OWNS
//   it (the data-om-exportable-video-with-duration-secs attribute, the
//   data-om-seek-to-time-frame listener, the svg/foreignObject wrapper,
//   and font inlining). NEVER put the exportable attribute on any other
//   element — a second "exportable root" makes the host timeline and the
//   video exporter bind to the wrong element, and playback control /
//   export silently break.
//
// HOW TIME WORKS: each OM_SCENES entry is a named slice of the authored
// timeline. CUES.Name is that section's authored start (the running sum of
// authored lengths, in literal order). useComposition().T is the authored
// clock: when the user trims or speeds a section on the host timeline, the
// engine replays that section's SAME authored slice over the new playback
// length — your choreography retimes, never cuts off. The optional "nat"
// field on an entry is the engine's authored-length anchor — the host
// timeline stamps it on the first retime; don't set it by hand.
//
// CUE-FIRST DISCIPLINE (what makes a piece read as one continuous video):
//   1. Write the OM_SCENES literal FIRST — it is the piece's outline.
//   2. One helper component per section for readability, but ALL of them
//      render ALL the time inside the one tree, keyed to CUES — never
//      conditionally mounted per section.
//   3. Define exactly three motion helpers up front (e.g.
//      MOTION = {enter, draw, pop} wrapping Easing curves) and use no
//      easing or transform outside them; one caption element, one visible
//      at a time (<Captions> has this built in).
//   A shared element that crosses a boundary is just motion whose start
//   and end straddle a cue: animate({from, to, start: CUES.Build - 0.4,
//   end: CUES.Build + 0.6})(T) glides through the boundary, and a user
//   slowing either section slows the glide without breaking it.
//
// RENDER FROM T ONLY: the exporter seeks each frame with a synchronous
// commit and may serialize the stage the moment the seek event returns —
// anything painted from useEffect or your own requestAnimationFrame lags
// that commit and exports stale. Render everything visible from T and this
// is automatic. A seeked frame is a deterministic render at that time.
//
// HARD CUTS are content now, not structure: wrap a shot's elements in
// <Shot from to> (visibility toggles at the cues; children stay mounted so
// images and videos hold their readiness). Shot also doubles as the
// perf gate for heavy far-away beats.
//
// LOOP SEAMS are the one surviving boundary rule: a looping piece shows
// its last authored frame immediately before its first — make them match
// (settle your choreography by authoredTotal, open it at 0).
//
// DIAGNOSTICS: choreography that references an unknown section name (a
// rename or deletion in OM_SCENES) shows a badge below the stage in the
// preview, outside the exportable svg — visible in preview screenshots,
// never in the exported video. An OM_SCENES section with no choreography
// keyed to it is a valid empty beat, not an error.
/* END USAGE */

// ─────────────────────────────────────────────────────────────────────────────

// ── Easing functions (hand-rolled, Popmotion-style) ─────────────────────────
// All easings take t ∈ [0,1] and return eased t ∈ [0,1] (may overshoot for back/elastic).
const Easing = {
  linear: t => t,
  // Quad
  easeInQuad: t => t * t,
  easeOutQuad: t => t * (2 - t),
  easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  // Cubic
  easeInCubic: t => t * t * t,
  easeOutCubic: t => --t * t * t + 1,
  easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  // Quart
  easeInQuart: t => t * t * t * t,
  easeOutQuart: t => 1 - --t * t * t * t,
  easeInOutQuart: t => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * --t * t * t * t,
  // Expo
  easeInExpo: t => t === 0 ? 0 : Math.pow(2, 10 * (t - 1)),
  easeOutExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: t => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    if (t < 0.5) return 0.5 * Math.pow(2, 20 * t - 10);
    return 1 - 0.5 * Math.pow(2, -20 * t + 10);
  },
  // Sine
  easeInSine: t => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine: t => Math.sin(t * Math.PI / 2),
  easeInOutSine: t => -(Math.cos(Math.PI * t) - 1) / 2,
  // Back (overshoot)
  easeOutBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInBack: t => {
    const c1 = 1.70158,
      c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },
  easeInOutBack: t => {
    const c1 = 1.70158,
      c2 = c1 * 1.525;
    return t < 0.5 ? Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2) / 2 : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },
  // Elastic
  easeOutElastic: t => {
    const c4 = 2 * Math.PI / 3;
    if (t === 0) return 0;
    if (t === 1) return 1;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  }
};

// ── Core interpolation helpers ──────────────────────────────────────────────

// Clamp a value to [min, max]
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// interpolate([0, 0.5, 1], [0, 100, 50], ease?) -> fn(t)
// Popmotion-style: linearly maps t across input keyframes to output values,
// with optional easing per segment (single fn or array of fns).
function interpolate(input, output, ease = Easing.linear) {
  return t => {
    if (t <= input[0]) return output[0];
    if (t >= input[input.length - 1]) return output[output.length - 1];
    for (let i = 0; i < input.length - 1; i++) {
      if (t >= input[i] && t <= input[i + 1]) {
        const span = input[i + 1] - input[i];
        const local = span === 0 ? 0 : (t - input[i]) / span;
        const easeFn = Array.isArray(ease) ? ease[i] || Easing.linear : ease;
        const eased = easeFn(local);
        return output[i] + (output[i + 1] - output[i]) * eased;
      }
    }
    return output[output.length - 1];
  };
}

// animate({from, to, start, end, ease})(t) — simpler single-segment tween.
// Returns `from` before `start`, `to` after `end`.
function animate({
  from = 0,
  to = 1,
  start = 0,
  end = 1,
  ease = Easing.easeInOutCubic
}) {
  return t => {
    if (t <= start) return from;
    if (t >= end) return to;
    const local = (t - start) / (end - start);
    return from + (to - from) * ease(local);
  };
}

// ── Timeline context ────────────────────────────────────────────────────────

const TimelineContext = React.createContext({
  time: 0,
  duration: 10,
  playing: false
});
const useTime = () => React.useContext(TimelineContext).time;
const useTimeline = () => React.useContext(TimelineContext);

// How long a marked (detail.playing === true) host seek keeps the
// external-playback latch alive with no successor. The host play bar's
// seek pump is one-in-flight/latest-wins, so its inter-seek gap is tens
// of milliseconds in the worst case — 400ms is far above that, so a
// marked stream that dies mid-play decays the latch promptly.
var SS_EXT_PLAY_MS = 400;

// ── Font inlining ───────────────────────────────────────────────────────────
// Copy every @font-face rule from the page into a <style> inside the svg's
// foreignObject, with font URLs rewritten to data: URLs. Makes the svg
// self-describing so serializing it alone (video export fast path) still
// renders with the right fonts. Sets data-om-fonts-inlined on the svg when
// done so the exporter can wait for it.

function useInlineFontsInto(svgRef) {
  React.useEffect(() => {
    const svg = svgRef.current;
    const host = svg && svg.querySelector('foreignObject > div');
    if (!svg || !host) return;
    let cancelled = false;
    (async () => {
      const rules = [];
      for (const ss of document.styleSheets) {
        let cssRules;
        try {
          cssRules = ss.cssRules;
        } catch {
          // Cross-origin sheet without crossorigin attr (e.g. the standard
          // fonts.googleapis.com <link>) — fetch the CSS text directly and
          // regex-extract the @font-face blocks.
          if (ss.href) {
            try {
              const txt = await fetch(ss.href).then(r => {
                if (!r.ok) throw 0;
                return r.text();
              });
              for (const ff of txt.match(/@font-face\s*{[^}]*}/g) || []) rules.push({
                css: ff,
                base: ss.href
              });
            } catch {}
          }
          continue;
        }
        if (!cssRules) continue;
        for (const r of cssRules) {
          if (r.type === CSSRule.FONT_FACE_RULE) {
            rules.push({
              css: r.cssText,
              base: ss.href || location.href
            });
          }
        }
      }
      const toDataURL = url => fetch(url).then(r => {
        if (!r.ok) throw 0;
        return r.blob();
      }).then(b => new Promise(res => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = () => res(url);
        fr.readAsDataURL(b);
      })).catch(() => url);
      const parts = await Promise.all(rules.map(async ({
        css,
        base
      }) => {
        const re = /url\((['"]?)([^'")]+)\1\)/g;
        let out = css,
          m;
        while (m = re.exec(css)) {
          const u = m[2];
          if (u.startsWith('data:')) continue;
          let abs;
          try {
            abs = new URL(u, base).href;
          } catch {
            continue;
          }
          out = out.split(m[0]).join(`url("${await toDataURL(abs)}")`);
        }
        return out;
      }));
      if (cancelled || !parts.length) {
        svg.setAttribute('data-om-fonts-inlined', 'true');
        return;
      }
      const style = document.createElement('style');
      style.textContent = parts.join('\n');
      host.insertBefore(style, host.firstChild);
      svg.setAttribute('data-om-fonts-inlined', 'true');
    })();
    return () => {
      cancelled = true;
    };
  }, []);
}
function Stage({
  width = 1280,
  height = 720,
  duration = 10,
  background = '#f6f4ef',
  fps = 60,
  loop = true,
  autoplay = true,
  // Parsed playback object ({mode:'loop'} | {mode:'times',count:N}) or
  // null. When present it overrides the legacy loop prop — CompositionStage
  // passes the validated value from the OM_PLAYBACK authoring contract.
  playback = null,
  persistKey = 'animstage-v3',
  children
}) {
  // Props arrive as strings when Stage is mounted via <x-import> (DC
  // projects) — coerce so style={{width}} gets a number React can px-ify.
  width = +width || 1280;
  height = +height || 720;
  duration = +duration || 10;
  fps = +fps || 60;
  if (typeof loop === 'string') loop = loop !== 'false';
  if (typeof autoplay === 'string') autoplay = autoplay !== 'false';
  const playTimes = playback && playback.mode === 'times' ? playback.count : null;
  const loopEff = playback ? playback.mode === 'loop' : loop;
  const [time, setTime] = React.useState(() => {
    try {
      const v = parseFloat(localStorage.getItem(persistKey + ':t') || '0');
      return isFinite(v) ? clamp(v, 0, duration) : 0;
    } catch {
      return 0;
    }
  });
  const [playing, setPlaying] = React.useState(autoplay);
  // The external-playback latch: true while the HOST play bar is driving
  // time forward as genuine continuous playback (its play-loop seeks
  // carry detail.playing === true). The engine's own clock stays paused
  // the whole time — exactly one clock ever drives — so this is a
  // separate bit, not a second meaning for `playing`. Set and cleared
  // in the seek handler below; decays via SS_EXT_PLAY_MS when the
  // marked stream stops without a parting unmarked seek.
  const [extPlay, setExtPlay] = React.useState(false);
  const extPlayTimerRef = React.useRef(null);
  const [hoverTime, setHoverTime] = React.useState(null);
  const [scale, setScale] = React.useState(1);
  const stageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const rafRef = React.useRef(null);
  const lastTsRef = React.useRef(null);

  // Persist playhead
  React.useEffect(() => {
    try {
      localStorage.setItem(persistKey + ':t', String(time));
    } catch {}
  }, [time, persistKey]);

  // Auto-scale to fit viewport
  React.useEffect(() => {
    if (!stageRef.current) return;
    const el = stageRef.current;
    const measure = () => {
      const barH = 44; // playback bar height
      const s = Math.min(el.clientWidth / width, (el.clientHeight - barH) / height);
      setScale(Math.max(0.05, s));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [width, height]);

  // Passes completed since playback last started. Lives in a ref so the
  // per-frame wrap can count without re-running this effect; reset on
  // every (re)start so a fresh play (or a host restart) gets the full
  // run count again.
  const passesRef = React.useRef(0);

  // Animation loop
  React.useEffect(() => {
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    passesRef.current = 0;
    const step = ts => {
      if (lastTsRef.current == null) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      setTime(t => {
        let next = t + dt;
        if (next >= duration) {
          if (playTimes !== null) {
            // Play N times then hold the last frame — the partial pass a
            // mid-timeline start produces counts as a pass, so the piece
            // never runs longer than N full durations.
            passesRef.current += 1;
            if (passesRef.current >= playTimes) {
              next = duration;
              setPlaying(false);
            } else {
              next = next % duration;
            }
          } else if (loopEff) {
            next = next % duration;
          } else {
            next = duration;
            setPlaying(false);
          }
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, duration, loopEff, playTimes]);

  // Keyboard: space = play/pause, ← → = seek
  React.useEffect(() => {
    const onKey = e => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        setPlaying(p => !p);
      } else if (e.code === 'ArrowLeft') {
        setTime(t => clamp(t - (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.code === 'ArrowRight') {
        setTime(t => clamp(t + (e.shiftKey ? 1 : 0.1), 0, duration));
      } else if (e.key === '0' || e.code === 'Home') {
        setTime(0);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [duration]);

  // Video-export protocol + the editor's play bar: hosts dispatch this
  // event per frame; pause + sync the playhead so the frame shows exactly
  // that timestamp. The host play bar marks its play-loop seeks with
  // detail.playing === true — the mark latches extPlay (playback is
  // playback even when a host clock drives it), while ANY unmarked seek
  // (scrub, step, export frame, the transport's pause park) clears the
  // latch in the same commit it retimes, so a seeked frame still renders
  // exactly one scene's state. The engine's own clock pauses either way.
  React.useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    // Sync-seek capability: a dispatcher that marks its seek with
    // detail.sync === true gets the commit applied via ReactDOM.flushSync,
    // so the stage DOM reflects the seeked frame the moment dispatchEvent
    // returns. The video exporter keys off the data-om-sync-seek
    // advertisement to drop its two-display-refresh settle (that wait only
    // exists to let React's async commit land — serialization needs the
    // committed DOM, not the paint). Feature-detected: a runtime without
    // ReactDOM.flushSync never advertises and every seek takes the async
    // path. Unmarked seeks (scrubs, the host play bar) stay async — a
    // forced sync render per pointermove would tax the editor for no one.
    const canSyncSeek = typeof ReactDOM !== 'undefined' && typeof ReactDOM.flushSync === 'function';
    const onSeek = e => {
      const apply = () => {
        setPlaying(false);
        const hostPlay = !!(e.detail && e.detail.playing === true);
        if (extPlayTimerRef.current) {
          clearTimeout(extPlayTimerRef.current);
          extPlayTimerRef.current = null;
        }
        if (hostPlay) {
          // Watchdog: the latch is only as alive as its seek stream. If the
          // host stops without a parting seek (tab jank, bar unmount), the
          // latch decays on its own rather than stranding extPlaying true.
          extPlayTimerRef.current = setTimeout(() => {
            extPlayTimerRef.current = null;
            setExtPlay(false);
          }, SS_EXT_PLAY_MS);
        }
        setExtPlay(hostPlay);
        setTime(clamp(e.detail.time, 0, duration));
      };
      // flushSync is safe here: a native DOM listener runs outside React's
      // lifecycle, and the exporter's dispatchEvent is synchronous, so the
      // commit lands in the same JS task — the engine's own rAF loop can
      // never interleave between seek and serialize.
      if (canSyncSeek && e.detail && e.detail.sync === true) {
        ReactDOM.flushSync(apply);
      } else {
        apply();
      }
    };
    el.addEventListener('data-om-seek-to-time-frame', onSeek);
    if (canSyncSeek) el.setAttribute('data-om-sync-seek', 'true');
    return () => {
      el.removeEventListener('data-om-seek-to-time-frame', onSeek);
      el.removeAttribute('data-om-sync-seek');
      if (extPlayTimerRef.current) {
        clearTimeout(extPlayTimerRef.current);
        extPlayTimerRef.current = null;
      }
      // Drop the latch too: this cleanup runs on every duration change
      // (an agent edit can retime mid-host-play, no gesture involved) and
      // the new effect instance arms no watchdog — clearing only the
      // timer could strand extPlay true forever if the marked stream died
      // in the gap. Fail toward cut: the next marked seek re-latches.
      setExtPlay(false);
    };
  }, [duration]);

  // Inline @font-face rules into the svg's foreignObject so the svg is
  // self-describing — serializing it alone (for video export) then renders
  // with the right fonts. Sets data-om-fonts-inlined once done.
  useInlineFontsInto(canvasRef);
  const displayTime = hoverTime != null ? hoverTime : time;
  const ctxValue = React.useMemo(
  // extPlaying is ADDITIVE: "time is advancing under an external
  // driver's continuous playback". `playing` keeps meaning the
  // engine's OWN clock — the hidden PlaybackBar glyph (and through it
  // the host's clock-reporter/adoption channel) reads that — and
  // CompositionClock is the one consumer that widens to either.
  () => ({
    time: displayTime,
    duration,
    playing,
    extPlaying: extPlay,
    setTime,
    setPlaying
  }), [displayTime, duration, playing, extPlay]);
  return (
    /*#__PURE__*/
    // data-om-starter: inert presence marker — Claude Design's starter-usage
    // probe reads it; it renders nothing. Keep it on this root element.
    React.createElement("div", {
      ref: stageRef,
      "data-om-starter": "animations-v3",
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: '#0a0a0a',
        fontFamily: 'Inter, system-ui, sans-serif'
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        flex: 1,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        minHeight: 0
      }
    }, /*#__PURE__*/React.createElement("svg", {
      ref: canvasRef,
      width: width,
      height: height,
      "data-om-exportable-video-with-duration-secs": duration,
      style: {
        transform: `scale(${scale})`,
        transformOrigin: 'center',
        flexShrink: 0,
        boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
        display: 'block'
      }
    }, /*#__PURE__*/React.createElement("foreignObject", {
      x: "0",
      y: "0",
      width: "100%",
      height: "100%"
    }, /*#__PURE__*/React.createElement("div", {
      xmlns: "http://www.w3.org/1999/xhtml",
      style: {
        width,
        height,
        background,
        position: 'relative',
        overflow: 'hidden'
      }
    }, /*#__PURE__*/React.createElement(TimelineContext.Provider, {
      value: ctxValue
    }, children))))), /*#__PURE__*/React.createElement(PlaybackBar, {
      time: displayTime,
      actualTime: time,
      duration: duration,
      playing: playing,
      onPlayPause: () => setPlaying(p => !p),
      onReset: () => {
        setTime(0);
      },
      onSeek: t => setTime(t),
      onHover: t => setHoverTime(t)
    }))
  );
}

// ── Playback bar ────────────────────────────────────────────────────────────
// Play/pause, return-to-begin, scrub track, time display.
// Uses fixed-width time fields so layout doesn't thrash.

function PlaybackBar({
  time,
  duration,
  playing,
  onPlayPause,
  onReset,
  onSeek,
  onHover
}) {
  const trackRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);
  const timeFromEvent = React.useCallback(e => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    return x * duration;
  }, [duration]);
  const onTrackMove = e => {
    if (!trackRef.current) return;
    const t = timeFromEvent(e);
    if (dragging) {
      onSeek(t);
    } else {
      onHover(t);
    }
  };
  const onTrackLeave = () => {
    if (!dragging) onHover(null);
  };
  const onTrackDown = e => {
    setDragging(true);
    const t = timeFromEvent(e);
    onSeek(t);
    onHover(null);
  };
  React.useEffect(() => {
    if (!dragging) return;
    const onUp = () => setDragging(false);
    const onMove = e => {
      if (!trackRef.current) return;
      const t = timeFromEvent(e);
      onSeek(t);
    };
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('mousemove', onMove);
    };
  }, [dragging, timeFromEvent, onSeek]);
  const pct = duration > 0 ? time / duration * 100 : 0;
  const fmt = t => {
    const total = Math.max(0, t);
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    const cs = Math.floor(total * 100 % 100);
    return `${String(m).padStart(1, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
  };
  const mono = 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace';
  return /*#__PURE__*/React.createElement("div", {
    "data-omelette-chrome": true,
    style: {
      // Slimmed to visually match the host editor bar's basic row (the
      // single-scrubber look): transport first, tighter metrics, quieter
      // chrome. Shown only outside the app — the host bar suppresses this
      // whenever it is present.
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 12px',
      background: 'rgba(20,20,20,0.92)',
      borderTop: '1px solid rgba(255,255,255,0.08)',
      width: '100%',
      maxWidth: 680,
      alignSelf: 'center',
      borderRadius: 6,
      color: '#f6f4ef',
      fontFamily: 'Inter, system-ui, sans-serif',
      userSelect: 'none',
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement(IconButton, {
    onClick: onPlayPause,
    title: "Play/pause (space)"
  }, playing ? /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("rect", {
    x: "3",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  }), /*#__PURE__*/React.createElement("rect", {
    x: "8",
    y: "2",
    width: "3",
    height: "10",
    fill: "currentColor"
  })) : /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2l9 5-9 5V2z",
    fill: "currentColor"
  }))), /*#__PURE__*/React.createElement(IconButton, {
    onClick: onReset,
    title: "Return to start (0)"
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M3 2v10M12 2L5 7l7 5V2z",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinejoin: "round",
    strokeLinecap: "round"
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'right',
      color: '#f6f4ef'
    }
  }, fmt(time)), /*#__PURE__*/React.createElement("div", {
    ref: trackRef,
    onMouseMove: onTrackMove,
    onMouseLeave: onTrackLeave,
    onMouseDown: onTrackDown,
    style: {
      flex: 1,
      height: 22,
      position: 'relative',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 4,
      background: 'rgba(255,255,255,0.12)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      width: `${pct}%`,
      height: 4,
      background: 'oklch(72% 0.12 250)',
      borderRadius: 2
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: `${pct}%`,
      top: '50%',
      width: 12,
      height: 12,
      marginLeft: -6,
      marginTop: -6,
      background: '#fff',
      borderRadius: 6,
      boxShadow: '0 2px 4px rgba(0,0,0,0.4)'
    }
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontFamily: mono,
      fontSize: 12,
      fontVariantNumeric: 'tabular-nums',
      width: 64,
      textAlign: 'left',
      color: 'rgba(246,244,239,0.55)'
    }
  }, fmt(duration)), typeof VideoEncoder !== 'undefined' && /*#__PURE__*/React.createElement(IconButton, {
    title: "Export video",
    onClick: () => window.parent.postMessage({
      type: 'omelette:request-video-export'
    }, '*')
  }, /*#__PURE__*/React.createElement("svg", {
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M7 2v7m0 0L4 6m3 3l3-3M2 12h10",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))));
}
function IconButton({
  children,
  onClick,
  title
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    title: title,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
      width: 24,
      height: 24,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: hover ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 5,
      color: '#f6f4ef',
      cursor: 'pointer',
      padding: 0,
      transition: 'background 120ms'
    }
  }, children);
}

// ── Scene-list plumbing ──────────────────────────────────────────────────
// Guest-side validation of a scene list (the engine's own inputs: the
// authored prop, and host-dispatched updates). Mirrors the host parser's
// shape rules and constants — keep in sync with parseTimelineScenes in
// apps/web/src/shared/timeline.ts (16KB raw cap, 50 entries, dur finite in
// (0, 300]); returns null on any violation.
function ssParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 16 * 1024) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length === 0 || parsed.length > 50) return null;
  for (var i = 0; i < parsed.length; i++) {
    var s = parsed[i];
    if (typeof s !== 'object' || s === null) return null;
    if (typeof s.name !== 'string' || typeof s.dur !== 'number') return null;
    if (!isFinite(s.dur) || s.dur <= 0 || s.dur > 300) return null;
  }
  return parsed;
}

// Guest-side validation of the playback value — mirrors the host parser
// (shared/timeline.ts parseTimelinePlayback): {"mode":"loop"} or
// {"mode":"times","count":1..99}, strict all-or-nothing, null otherwise.
// Callers treat null as the loop default.
function ppParse(raw) {
  if (typeof raw !== 'string' || !raw || raw.length > 256) return null;
  var parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
  var keys = Object.keys(parsed);
  if (parsed.mode === 'loop') return keys.length === 1 ? {
    mode: 'loop'
  } : null;
  if (parsed.mode === 'times') {
    if (keys.length !== 2) return null;
    var c = parsed.count;
    if (typeof c !== 'number' || c !== Math.floor(c) || c < 1 || c > 99) return null;
    return {
      mode: 'times',
      count: c
    };
  }
  return null;
}

// Stamps the playback attribute VERBATIM from the authored raw string (the
// host's write-back anchors on that exact value) and listens for the
// host's post-write update event. Same shape as SceneSync; only rendered
// when the document authors a playback literal — an absent contract means
// the attribute stays absent and the document plays its default.
function PlaybackSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-playback', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      if (ppParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-playback-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-playback-update', onEvent);
      root.removeAttribute('data-om-timeline-playback');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// Renders inside the Stage (so it can reach the exportable root via
// closest()): stamps the scenes attribute VERBATIM from the current raw
// string — the host's write-back anchors on that exact value — and listens
// for the host's post-write update event.
function SceneSync(props) {
  var ref = React.useRef(null);
  var raw = props.raw;
  var onUpdate = props.onUpdate;
  React.useEffect(function () {
    var el = ref.current;
    if (!el) return;
    var root = el.closest('[data-om-exportable-video-with-duration-secs]');
    if (!root) return;
    root.setAttribute('data-om-timeline-scenes', raw);
    var onEvent = function (e) {
      var next = e && e.detail;
      // Ignore anything that doesn't validate — a bad update must not tear
      // down a working composition.
      if (ssParse(next)) onUpdate(next);
    };
    root.addEventListener('data-om-timeline-scenes-update', onEvent);
    return function () {
      root.removeEventListener('data-om-timeline-scenes-update', onEvent);
      root.removeAttribute('data-om-timeline-scenes');
    };
  }, [raw, onUpdate]);
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    style: {
      display: 'none'
    }
  });
}

// ── Continuous composition ──────────────────────────────────────────────

var CompositionContext = React.createContext(null);
function useComposition() {
  var ctx = React.useContext(CompositionContext);
  if (!ctx) throw new Error('useComposition() must be called inside <CompositionStage>');
  return ctx;
}
function ccDerive(scenes) {
  var playStart = 0;
  var authStart = 0;
  var sections = [];
  var table = Object.create(null);
  for (var i = 0; i < scenes.length; i++) {
    var s = scenes[i];
    var nat = typeof s.nat === 'number' && isFinite(s.nat) && s.nat > 0 ? s.nat : s.dur;
    sections.push({
      name: s.name,
      playStart: playStart,
      dur: s.dur,
      authStart: authStart,
      nat: nat
    });
    if (!Object.prototype.hasOwnProperty.call(table, s.name)) {
      table[s.name] = Math.round(authStart * 1000) / 1000;
    }
    playStart += s.dur;
    authStart += nat;
  }
  return {
    sections: sections,
    table: table,
    total: Math.round(playStart * 1000) / 1000,
    authoredTotal: Math.round(authStart * 1000) / 1000
  };
}
function ccWarp(d, t) {
  var ss = d.sections;
  if (ss.length === 0) return 0;
  var idx = ss.length - 1;
  for (var i = 0; i < ss.length; i++) {
    if (t < ss[i].playStart + ss[i].dur) {
      idx = i;
      break;
    }
  }
  var s = ss[idx];
  var local = Math.min(Math.max(t - s.playStart, 0), s.dur);
  var T = s.authStart + (s.dur > 0 ? local * (s.nat / s.dur) : 0);
  return Math.min(T, d.authoredTotal);
}
var CC_META = Object.assign(Object.create(null), {
  toString: 1,
  toLocaleString: 1,
  valueOf: 1,
  toJSON: 1,
  then: 1,
  constructor: 1,
  hasOwnProperty: 1,
  isPrototypeOf: 1,
  propertyIsEnumerable: 1,
  default: 1
});
function ccCueProxy(table, unknownRef) {
  if (typeof Proxy !== 'function') return table;
  return new Proxy(table, {
    get: function (target, prop) {
      if (typeof prop !== 'string' || prop in target) return target[prop];
      if (CC_META[prop] || prop.indexOf('@@') === 0) return Object.prototype[prop];
      unknownRef.current[prop] = true;
      return NaN;
    }
  });
}
function CcUnknownWatch(props) {
  var tl = useTimeline();
  React.useEffect(function () {
    var next = Object.keys(props.unknownRef.current).sort().join(', ');
    if (next !== props.badge) props.setBadge(next);
  }, [tl.time]);
  return null;
}
function CompositionClock(props) {
  var tl = useTimeline();
  var d = props.derived;
  var T = ccWarp(d, tl.time);
  var value = React.useMemo(function () {
    return {
      T: T,
      CUES: props.cues,
      time: tl.time,
      duration: tl.duration,
      authoredTotal: d.authoredTotal,
      playing: tl.playing || tl.extPlaying === true
    };
  }, [T, props.cues, tl.time, tl.duration, d, tl.playing, tl.extPlaying]);
  return /*#__PURE__*/React.createElement(CompositionContext.Provider, {
    value: value
  }, props.children);
}
function Shot(props) {
  var c = useComposition();
  var from = +props.from;
  var to = props.to == null ? Infinity : +props.to;
  var on = isFinite(from) && c.T >= from && c.T < to;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      visibility: on ? 'visible' : 'hidden'
    }
  }, props.children);
}
var CAPTION_FADE = 0.18;
function Captions(props) {
  var c = useComposition();
  var t = c.T;
  var items = (props.items || []).filter(function (it) {
    return it && isFinite(+it.at);
  }).sort(function (a, b) {
    return a.at - b.at;
  });
  var active = null;
  var end = Infinity;
  for (var i = 0; i < items.length; i++) {
    if (t < items[i].at) break;
    active = items[i];
    end = typeof active.until === 'number' && isFinite(active.until) ? active.until : i + 1 < items.length ? items[i + 1].at : Infinity;
  }
  if (!active || t >= end) return null;
  var o = Math.min(1, (t - active.at) / CAPTION_FADE);
  if (isFinite(end)) o = Math.min(o, (end - t) / CAPTION_FADE);
  o = Math.max(0, Math.min(1, o));
  return /*#__PURE__*/React.createElement("div", {
    "data-om-caption": true,
    style: Object.assign({
      position: 'absolute',
      left: '8%',
      right: '8%',
      bottom: '7%',
      textAlign: 'center',
      opacity: o,
      pointerEvents: 'none',
      font: '500 30px Inter, system-ui, sans-serif',
      color: '#f6f4ef',
      textShadow: '0 1px 14px rgba(0,0,0,0.45)'
    }, props.style)
  }, active.text);
}
function CompositionStage(props) {
  var width = +props.width || 1280;
  var height = +props.height || 720;
  var bg = props.bg || '#0b0b0e';
  var autoplay = props.autoplay == null ? true : String(props.autoplay) !== 'false';
  var loop = props.loop == null ? true : String(props.loop) !== 'false';
  var state = React.useState(props.scenes);
  var raw = state[0];
  var setRaw = state[1];
  var scenes = React.useMemo(function () {
    return ssParse(raw);
  }, [raw]);
  var pstate = React.useState(props.playback);
  var praw = pstate[0];
  var setPraw = pstate[1];
  var pb = React.useMemo(function () {
    return ppParse(praw);
  }, [praw]);
  var unknownRef = React.useRef({});
  var badgeState = React.useState('');
  var badge = badgeState[0];
  var setBadge = badgeState[1];
  var derived = React.useMemo(function () {
    unknownRef.current = {};
    return scenes ? ccDerive(scenes) : null;
  }, [scenes]);
  var cues = React.useMemo(function () {
    return derived ? ccCueProxy(derived.table, unknownRef) : null;
  }, [derived]);
  React.useEffect(function () {
    var next = Object.keys(unknownRef.current).sort().join(', ');
    if (next !== badge) setBadge(next);
  });
  if (!scenes) {
    return /*#__PURE__*/React.createElement("div", {
      style: {
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0b0b0e',
        color: '#c96442',
        font: '500 16px Inter, system-ui, sans-serif',
        textAlign: 'center'
      }
    }, "animations-v3: the scenes prop isn't a valid JSON scene list", /*#__PURE__*/React.createElement("br", null), "(expected '[", '{', "\"name\":\"\u2026\",\"dur\":N", '}', ", \u2026]')");
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Stage, {
    width: width,
    height: height,
    duration: derived.total,
    background: bg,
    autoplay: autoplay,
    loop: loop,
    playback: pb
  }, /*#__PURE__*/React.createElement(SceneSync, {
    raw: raw,
    onUpdate: setRaw
  }), typeof praw === 'string' && praw !== '' && /*#__PURE__*/React.createElement(PlaybackSync, {
    raw: praw,
    onUpdate: setPraw
  }), /*#__PURE__*/React.createElement(CompositionClock, {
    derived: derived,
    cues: cues
  }, props.children), /*#__PURE__*/React.createElement(CcUnknownWatch, {
    unknownRef: unknownRef,
    badge: badge,
    setBadge: setBadge
  })), badge !== '' &&
  /*#__PURE__*/
  // Sibling of Stage, outside the exportable <svg>: visible in the
  // preview (and its screenshots), never in the exported video.
  React.createElement("div", {
    "data-om-unknown-cues": true,
    style: {
      position: 'absolute',
      left: 12,
      bottom: 56,
      zIndex: 10,
      padding: '6px 10px',
      borderRadius: 6,
      background: 'rgba(0,0,0,0.72)',
      color: '#e8906a',
      font: '500 12px Inter, system-ui, sans-serif',
      pointerEvents: 'none'
    }
  }, "choreography references unknown section", badge.indexOf(',') >= 0 ? 's' : '', ": ", badge));
}

// Strokes as layers: paint multiplies, so stroke images stacked with
// mix-blend-mode:multiply over the paper reproduce the flat render.

var WC_PIXEL_CAP = 11000000;
function wcLayerOpts(props) {
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || 1;
  return {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(WC_PIXEL_CAP / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    quality: props.quality == null ? undefined : +props.quality
  };
}
var wcWarned = {};
function wcWarnOnce(key, message, err) {
  if (wcWarned[key]) return;
  wcWarned[key] = true;
  console.warn(message, err);
}
function useWatercolorLayers(painting, opts) {
  var kit = window.WatercolorKit;
  if (typeof painting !== 'function' || !kit || typeof kit.layers !== 'function') return null;
  try {
    return kit.layers(painting, wcLayerOpts(opts || {}));
  } catch (e) {
    wcWarnOnce('layers:' + e, 'watercolor painting failed to build; rendering the fallback sheet', e);
    return null;
  }
}
var WatercolorSheetContext = React.createContext(null);
function WatercolorSheet(props) {
  var L = props.layers || null;
  var style = Object.assign({
    position: 'relative',
    display: 'block',
    width: '100%',
    aspectRatio: L ? L.width + ' / ' + L.height : '3 / 4',
    isolation: 'isolate',
    overflow: 'hidden'
  }, props.style);
  if (!L) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign(style, {
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      })
    }, "watercolor-kit.js not loaded (or the painting failed to build)");
  }
  return /*#__PURE__*/React.createElement(WatercolorSheetContext.Provider, {
    value: L
  }, /*#__PURE__*/React.createElement("div", {
    style: style,
    "data-om-watercolor-sheet": true
  }, /*#__PURE__*/React.createElement("img", {
    src: L.paper,
    alt: props.alt || '',
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      width: '100%',
      height: '100%',
      display: 'block'
    }
  }), props.children));
}
function WatercolorStroke(props) {
  var fromSheet = React.useContext(WatercolorSheetContext);
  var L = props.layers || fromSheet;
  if (!L) return null;
  var i = +props.index;
  if (!(i >= 0) || i >= L.count) return null;
  var at = props.at == null ? 1 : clamp(+props.at, 0, 1);
  if (!(at > 0)) return null;
  var box, src;
  try {
    box = L.box(i);
    src = box ? L.src(i, at) : null;
  } catch (e) {
    wcWarnOnce('stroke:' + i + ':' + e, 'watercolor stroke ' + i + ' failed to render; skipping it', e);
    return null;
  }
  if (!box || !src) return null;
  var style = Object.assign({
    position: 'absolute',
    display: 'block',
    left: box.x * 100 + '%',
    top: box.y * 100 + '%',
    width: box.w * 100 + '%',
    height: box.h * 100 + '%',
    mixBlendMode: L.kind(i) === 'reserve' ? 'normal' : 'multiply',
    pointerEvents: 'none'
  }, props.style);
  return /*#__PURE__*/React.createElement("img", {
    src: src,
    alt: "",
    "data-om-watercolor-stroke": i,
    "data-om-stroke-kind": L.kind(i),
    style: style
  });
}

// The default watercolor moment: the painting assembled from its strokes,
// each appearing in painting order (a pure function of T).
function WatercolorPainting(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var eased = Easing.easeInOutQuad(u);
  var L = useWatercolorLayers(props.painting, props);
  var tick = React.useState(0)[1];
  var warmed = React.useRef(null);
  React.useEffect(function () {
    if (!L || typeof L.warm !== 'function') return;
    var p = L.warm();
    if (warmed.current === p) return;
    var live = true;
    p.then(function () {
      warmed.current = p;
      if (live) tick(function (x) {
        return x + 1;
      });
    });
    return function () {
      live = false;
    };
  }, [L && L.paper, props.painting]);
  var strokes = [];
  if (L) {
    for (var i = 0; i < L.count; i++) {
      var sp = L.span(i);
      var at = clamp((eased - sp.from) / Math.max(sp.to - sp.from, 1e-6), 0, 1);
      if (at <= 0) break;
      strokes.push(/*#__PURE__*/React.createElement(WatercolorStroke, {
        key: i,
        layers: L,
        index: i,
        at: at
      }));
    }
  }
  return /*#__PURE__*/React.createElement(WatercolorSheet, {
    layers: L,
    style: props.style,
    alt: props.alt
  }, strokes);
}

// Paint-on watercolor reveal as a pure function of T — an <img> with a data:
// URL (the exporter serializes those as-is; a live canvas would export blank).
function WatercolorReveal(props) {
  var c = useComposition();
  var from = +props.from || 0;
  var to = props.to == null ? from + 6 : +props.to;
  var u = clamp((c.T - from) / Math.max(to - from, 0.001), 0, 1);
  var style = Object.assign({
    display: 'block',
    width: '100%',
    height: '100%',
    objectFit: 'contain'
  }, props.style);
  var frames = Array.isArray(props.frames) && props.frames.length ? props.frames : null;
  var steps = frames ? frames.length - 1 : Math.max(1, Math.round(+props.steps || 36));
  var i = Math.min(steps, Math.round(Easing.easeInOutQuad(u) * steps));
  var painting = typeof props.painting === 'function' ? props.painting : null;
  var kit = window.WatercolorKit;
  var w = +props.width || 900,
    h = +props.height || 1200;
  var askScale = +props.scale || Math.min(2, window.devicePixelRatio || 1);
  var opts = {
    width: w,
    height: h,
    scale: Math.min(askScale, Math.sqrt(11000000 / (w * h))),
    seed: props.seed == null ? undefined : +props.seed,
    steps: steps,
    type: props.format || 'image/jpeg',
    quality: props.quality == null ? 0.88 : +props.quality
  };
  var key = opts.width + 'x' + opts.height + '#' + opts.seed + '@' + opts.scale + '/' + steps + ':' + opts.type + '/' + opts.quality;
  var cache = React.useRef({
    fn: null,
    key: '',
    frames: {},
    baking: false
  }).current;
  var tick = React.useState(0)[1];
  if (cache.fn !== painting && String(cache.fn) !== String(painting) || cache.key !== key) {
    cache.key = key;
    cache.frames = {};
    cache.baking = false;
  }
  cache.fn = painting;
  React.useEffect(function () {
    if (frames || cache.baking || !painting || !kit || typeof kit.bake !== 'function') return;
    cache.baking = true;
    var target = cache.frames;
    try {
      kit.bake(painting, opts, function (n, _t, url) {
        target[n] = url;
      }).then(function (all) {
        if (cache.frames !== target) return;
        for (var n = 0; n < all.length; n++) target[n] = all[n];
        tick(function (x) {
          return x + 1;
        });
      }).catch(function () {
        /* failed bake: the guarded lazy path below still renders */
      });
    } catch (e) {
      /* oversized painting: the guarded lazy path below still renders */
    }
  });
  if (frames) return /*#__PURE__*/React.createElement("img", {
    src: frames[i],
    alt: props.alt || '',
    style: style
  });
  if (!kit || !painting) {
    return /*#__PURE__*/React.createElement("div", {
      style: Object.assign({
        width: '100%',
        height: '100%',
        background: '#f4f1e8',
        color: '#8a8270',
        font: '12px system-ui, sans-serif',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }, props.style)
    }, "watercolor-kit.js not loaded (or no painting function)");
  }
  if (!cache.frames[i]) {
    try {
      cache.frames[i] = kit.frame(painting, Object.assign({}, opts, {
        at: i / steps
      }));
    } catch (e) {
      return /*#__PURE__*/React.createElement("div", {
        style: Object.assign({
          width: '100%',
          height: '100%',
          background: '#f4f1e8',
          color: '#8a8270',
          font: '12px system-ui, sans-serif',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }, props.style)
      }, "painting too large to render (", String(e && e.message).slice(0, 80), ")");
    }
  }
  return /*#__PURE__*/React.createElement("img", {
    src: cache.frames[i],
    alt: props.alt || '',
    style: style
  });
}
Object.assign(window, {
  Easing,
  interpolate,
  animate,
  clamp,
  TimelineContext,
  useTime,
  useTimeline,
  Stage,
  PlaybackBar,
  CompositionStage,
  useComposition,
  Shot,
  Captions,
  WatercolorReveal,
  WatercolorPainting,
  WatercolorSheet,
  WatercolorStroke,
  useWatercolorLayers
});

})();
;(function(){ /* ── axolotl-river.jsx ── */
/* Underwater riverscape + axolotl assistant — continuous composition. */
const useComposition = window.useComposition;
const interpolate = window.interpolate;
const animate = window.animate;
const clamp = window.clamp || ((v, a, b) => Math.min(b, Math.max(a, v)));
const Easing = window.Easing || {
  easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  easeOutBack: t => 1 + 2.70158 * Math.pow(t - 1, 3) + 1.70158 * Math.pow(t - 1, 2),
  easeOutQuart: t => 1 - Math.pow(1 - t, 4)
};
const W = 1080,
  H = 1080;
const CX = W / 2,
  CY = H / 2;

/* three motion helpers, nothing else eases */
const MOTION = {
  glide: Easing.easeInOutCubic,
  pop: Easing.easeOutBack,
  settle: Easing.easeOutQuart
};
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const lerp = (a, b, t) => a + (b - a) * t;

/* deterministic scenery -------------------------------------------------- */
const R = rng(20260912);
const SILT = Array.from({
  length: 90
}, () => ({
  x: -600 + R() * 3100,
  y: -200 + R() * 1500,
  r: 1.5 + R() * 5,
  o: 0.12 + R() * 0.4,
  k: 1 + Math.floor(R() * 3),
  ph: R(),
  amp: 14 + R() * 60
}));
const ROCKS = Array.from({
  length: 18
}, (_, i) => ({
  x: -900 + i * 200 + R() * 90,
  y: 880 + R() * 150,
  rx: 90 + R() * 190,
  ry: 40 + R() * 90,
  tone: R()
}));
const REEDS = Array.from({
  length: 38
}, (_, i) => {
  const short = R() < 0.45;
  return {
    x: -1000 + i * 98 + R() * 60,
    base: 1048 + R() * 74,
    h: short ? 160 + R() * 190 : 340 + R() * 520,
    w: short ? 30 + R() * 26 : 19 + R() * 22,
    ph: R(),
    hue: 152 + R() * 38
  };
});
const RAYS = Array.from({
  length: 8
}, (_, i) => ({
  x: -300 + i * 300 + R() * 120,
  w: 90 + R() * 190,
  o: 0.05 + R() * 0.09,
  sk: -26 + R() * 16,
  ph: R()
}));
const FISH = Array.from({
  length: 5
}, () => ({
  y: 260 + R() * 520,
  s: 0.5 + R() * 0.9,
  o: 0.28 + R() * 0.3,
  ph: R(),
  dir: R() > 0.5 ? 1 : -1,
  bob: 10 + R() * 26
}));

/* camera ----------------------------------------------------------------- */
function cameraAt(T, C, total, fit = 0.6) {
  const F = v => v > 1.6 ? 1.6 + (v - 1.6) * fit : v;
  const wob = Math.sin(2 * Math.PI * 2 * (T / total));
  const idle = (n, x, y, s2) => ({
    t: C['Idle' + n] + 0.5,
    x,
    y,
    s: F(s2)
  });
  const track = [{
    t: 0,
    x: -150,
    y: 558,
    s: 0.72
  }, {
    t: C.River + 2.2,
    x: 20,
    y: 550,
    s: 0.78
  }, {
    t: C.Approach,
    x: 420,
    y: 546,
    s: 0.98
  }, {
    t: C.Attentive + 0.4,
    x: 1150,
    y: 572,
    s: F(2.85)
  }, idle(1, 1150, 568, 2.72), {
    t: C.Thinking + 0.8,
    x: 1140,
    y: 564,
    s: F(2.9)
  }, idle(2, 1152, 568, 2.72), {
    t: C.Idea + 0.5,
    x: 1150,
    y: 576,
    s: F(2.8)
  }, idle(3, 1148, 568, 2.72), {
    t: C.Confused + 0.8,
    x: 1160,
    y: 568,
    s: F(2.88)
  }, idle(4, 1152, 568, 2.72), {
    t: C.Focused + 0.8,
    x: 1150,
    y: 570,
    s: F(2.9)
  }, idle(5, 1150, 568, 2.72), {
    t: C.Sleepy + 1.2,
    x: 1148,
    y: 578,
    s: F(2.82)
  }, idle(6, 1150, 570, 2.76), {
    t: C.Pullback,
    x: 1150,
    y: 572,
    s: F(2.7)
  }, {
    t: total,
    x: -150,
    y: 558,
    s: 0.72
  }];
  let i = 0;
  while (i < track.length - 2 && T >= track[i + 1].t) i++;
  const p = clamp((T - track[i].t) / Math.max(0.001, track[i + 1].t - track[i].t), 0, 1);
  return {
    x: spline(track, k => k.x, i, p) + wob * 6,
    y: spline(track, k => k.y, i, p) + wob * 3,
    s: Math.exp(spline(track, k => Math.log(k.s), i, p))
  };
}

/* monotone cubic Hermite: continuous velocity across keyframes, no overshoot */
function spline(track, get, i, p) {
  const n = track.length;
  const at = k => clamp(k, 0, n - 1);
  const slope = k => {
    const h = Math.max(0.001, track[at(k + 1)].t - track[at(k)].t);
    return (get(track[at(k + 1)]) - get(track[at(k)])) / h;
  };
  const tangent = k => {
    if (k <= 0 || k >= n - 1) return 0;
    const d0 = slope(k - 1),
      d1 = slope(k);
    if (d0 * d1 <= 0) return 0;
    const m = (d0 + d1) / 2;
    const lim = 3 * Math.min(Math.abs(d0), Math.abs(d1));
    return Math.sign(m) * Math.min(Math.abs(m), lim);
  };
  const h = Math.max(0.001, track[i + 1].t - track[i].t);
  const v0 = get(track[i]),
    v1 = get(track[i + 1]);
  const m0 = tangent(i),
    m1 = tangent(i + 1);
  const p2 = p * p,
    p3 = p2 * p;
  return (2 * p3 - 3 * p2 + 1) * v0 + (p3 - 2 * p2 + p) * h * m0 + (-2 * p3 + 3 * p2) * v1 + (p3 - p2) * h * m1;
}

/* expressions ------------------------------------------------------------ */
const REST = {
  browL: 0,
  browR: 0,
  rotL: 0,
  rotR: 0,
  lid: 0.02,
  eyeX: 0,
  eyeY: 0,
  eyeSq: 1,
  curve: 10,
  open: 0,
  mw: 1,
  mrot: 0,
  gill: 0,
  tilt: 0,
  bob: 0,
  spark: 0,
  blush: 0,
  zzz: 0
};
const E = o => ({
  ...REST,
  ...o
});
const IDLE = E({
  curve: 15,
  gill: 2,
  lid: 0.06,
  tilt: 1.5,
  bob: -2
});
function exprTrack(C, total) {
  const idle = (n, extra) => ({
    t: C['Idle' + n] + 0.5,
    v: extra ? {
      ...IDLE,
      ...extra
    } : IDLE
  });
  return [{
    t: 0,
    v: REST
  }, {
    t: C.Approach + 1.0,
    v: E({
      curve: 12,
      gill: 3
    })
  }, {
    t: C.Attentive + 0.7,
    v: E({
      lid: 0,
      eyeSq: 1.08,
      curve: 22,
      gill: 10,
      tilt: -1.5,
      browL: -5,
      browR: -5
    })
  }, idle(1), {
    t: C.Thinking + 1.1,
    v: E({
      eyeX: -13,
      eyeY: -14,
      browL: -9,
      browR: -3,
      rotL: -9,
      rotR: 5,
      curve: -4,
      mw: 0.78,
      mrot: -5,
      gill: -5,
      tilt: 5,
      lid: 0.2
    })
  }, idle(2, {
    tilt: -1.5
  }), {
    t: C.Idea + 0.55,
    v: E({
      eyeSq: 1.18,
      lid: -0.05,
      curve: 26,
      open: 30,
      gill: 16,
      bob: -22,
      spark: 1,
      blush: 0.5,
      browL: -14,
      browR: -14
    })
  }, {
    t: C.Idea + 1.5,
    v: E({
      eyeSq: 1.1,
      curve: 24,
      open: 12,
      gill: 12,
      bob: -8,
      spark: 0.35,
      blush: 0.3
    })
  }, idle(3, {
    blush: 0.12
  }), {
    t: C.Confused + 1.0,
    v: E({
      rotL: 16,
      rotR: -18,
      browL: -11,
      browR: 5,
      eyeSq: 0.94,
      eyeX: 7,
      curve: -9,
      mrot: 9,
      mw: 0.85,
      tilt: 8,
      gill: -3
    })
  }, idle(4, {
    tilt: 3
  }), {
    t: C.Focused + 1.0,
    v: E({
      lid: 0.42,
      eyeSq: 0.8,
      browL: 7,
      browR: 7,
      rotL: -6,
      rotR: 6,
      curve: 1,
      mw: 1.12,
      gill: 4,
      tilt: -2.5
    })
  }, idle(5, {
    lid: 0.12
  }), {
    t: C.Sleepy + 0.9,
    v: E({
      lid: 0.66,
      eyeY: 7,
      curve: 6,
      open: 36,
      mw: 0.9,
      gill: -9,
      tilt: 4,
      bob: 6,
      zzz: 0.6
    })
  }, {
    t: C.Sleepy + 2.1,
    v: E({
      lid: 0.99,
      eyeY: 8,
      curve: 9,
      open: 4,
      gill: -11,
      tilt: 5,
      bob: 10,
      zzz: 1
    })
  }, idle(6, {
    lid: 0.3,
    zzz: 0.35,
    tilt: 2.5
  }), {
    t: C.Pullback + 1.4,
    v: E({
      lid: 0.5,
      curve: 14,
      gill: -2,
      tilt: 1,
      zzz: 0.4
    })
  }, {
    t: total,
    v: REST
  }];
}
function exprAt(T, C, total) {
  const tr = exprTrack(C, total);
  let i = 0;
  while (i < tr.length - 2 && T >= tr[i + 1].t) i++;
  const a = tr[i],
    b = tr[i + 1];
  const e = MOTION.glide(clamp((T - a.t) / Math.max(0.001, b.t - a.t), 0, 1));
  const out = {};
  for (const k in REST) out[k] = lerp(a.v[k], b.v[k], e);
  return out;
}

/* layer wrapper ---------------------------------------------------------- */
function Layer({
  depth,
  cam,
  blurBase = 0,
  children,
  z = 0
}) {
  const s = Math.pow(cam.s, depth);
  const blur = clamp(blurBase + Math.abs(1 - depth) * (cam.s - 1) * 1.7, 0, 16);
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      zIndex: z,
      transform: `translate(${CX}px, ${CY}px) scale(${s}) translate(${-cam.x}px, ${-cam.y}px)`,
      transformOrigin: '0 0',
      filter: blur > 0.4 ? `blur(${blur}px)` : 'none',
      willChange: 'transform'
    }
  }, children);
}

/* scenery components ----------------------------------------------------- */
function WaterBody({
  ph
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -900,
      top: -500,
      width: 3800,
      height: 2200
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      background: 'linear-gradient(#0c4a52 0%, #0a3f4c 26%, #093642 55%, #072b36 78%, #05202a 100%)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      top: 0,
      width: '100%',
      height: 760,
      background: 'radial-gradient(120% 100% at 42% -10%, rgba(178,235,225,0.5) 0%, rgba(120,205,200,0.16) 38%, rgba(10,60,70,0) 72%)',
      transform: `translateX(${Math.sin(ph) * 40}px)`
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: 0,
      top: 440,
      width: '100%',
      height: 900,
      background: 'radial-gradient(90% 100% at 78% 40%, rgba(6,42,56,0.75) 0%, rgba(6,42,56,0) 70%)'
    }
  }));
}
function Surface({
  ph
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -900,
      top: -560,
      width: 3800,
      height: 620,
      overflow: 'hidden'
    }
  }, [0, 1, 2].map(i => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 380 + i * 46,
      height: 90,
      background: `radial-gradient(60% 100% at 50% 100%, rgba(214,250,245,${0.16 - i * 0.045}) 0%, rgba(214,250,245,0) 70%)`,
      transform: `translateX(${Math.sin(ph + i * 1.3) * (110 - i * 30)}px) scaleY(${1 + Math.sin(ph * 1.6 + i) * 0.18})`
    }
  })));
}
function Rays({
  ph
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -700,
      top: -420,
      width: 3400,
      height: 1900
    }
  }, RAYS.map((r, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      position: 'absolute',
      left: r.x + 700,
      top: 0,
      width: r.w,
      height: 1750,
      background: `linear-gradient(rgba(206,248,240,${r.o}) 0%, rgba(206,248,240,${r.o * 0.5}) 42%, rgba(206,248,240,0) 88%)`,
      transform: `skewX(${r.sk + Math.sin(ph + r.ph * 6.28) * 2.4}deg) scaleX(${1 + Math.sin(ph * 1.4 + r.ph * 6.28) * 0.16})`,
      transformOrigin: '50% 0',
      borderRadius: '50% / 8%'
    }
  })));
}
function Rocks({
  ph
}) {
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "-600 700 3200 700",
    style: {
      position: 'absolute',
      left: -600,
      top: 700,
      width: 3200,
      height: 700,
      overflow: 'visible'
    }
  }, ROCKS.map((r, i) => /*#__PURE__*/React.createElement("ellipse", {
    key: i,
    cx: r.x,
    cy: r.y + Math.sin(ph + i) * 2,
    rx: r.rx,
    ry: r.ry,
    fill: `rgb(${12 + r.tone * 14}, ${38 + r.tone * 16}, ${44 + r.tone * 18})`,
    opacity: 0.95
  })), /*#__PURE__*/React.createElement("rect", {
    x: -1600,
    y: 1040,
    width: 5200,
    height: 500,
    fill: "#07222c"
  }));
}
function Reeds({
  ph,
  near
}) {
  const set = REEDS.filter((_, i) => near ? i % 3 === 0 : i % 3 !== 0);
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "-600 400 3200 900",
    style: {
      position: 'absolute',
      left: -600,
      top: 400,
      width: 3200,
      height: 900,
      overflow: 'visible'
    }
  }, set.map((rd, i) => {
    const sway = Math.sin(ph * 1.1 + rd.ph * 6.28) * (7 + rd.h / 70);
    const lean = (rd.ph - 0.5) * 200;
    const w = rd.w;
    const p1x = rd.x + lean * 0.18 + sway,
      p1y = rd.base - rd.h * 0.45;
    const p2x = rd.x + lean * 0.62 + sway * 2.2,
      p2y = rd.base - rd.h * 0.74;
    const p3x = rd.x + lean + sway * 3.2,
      p3y = rd.base - rd.h;
    const L = near ? 15 : 23;
    return /*#__PURE__*/React.createElement("g", {
      key: i,
      opacity: near ? 0.94 : 0.86
    }, /*#__PURE__*/React.createElement("path", {
      d: `M ${rd.x - w / 2} ${rd.base} C ${p1x - w * 0.34} ${p1y} ${p2x - w * 0.15} ${p2y} ${p3x} ${p3y} C ${p2x + w * 0.15} ${p2y} ${p1x + w * 0.34} ${p1y} ${rd.x + w / 2} ${rd.base} Z`,
      fill: `hsl(${rd.hue} 40% ${L}%)`,
      stroke: `hsl(${rd.hue} 46% ${L - 6}%)`,
      strokeWidth: "2.5"
    }), /*#__PURE__*/React.createElement("path", {
      d: `M ${rd.x - w * 0.22} ${rd.base} C ${p1x - w * 0.22} ${p1y} ${p2x - w * 0.1} ${p2y} ${p3x} ${p3y}`,
      stroke: `hsl(${rd.hue + 8} 30% ${L + 14}%)`,
      strokeWidth: Math.max(2.5, w * 0.18),
      fill: "none",
      strokeLinecap: "round",
      opacity: "0.5"
    }), /*#__PURE__*/React.createElement("path", {
      d: `M ${rd.x} ${rd.base} C ${p1x} ${p1y} ${p2x} ${p2y} ${p3x} ${p3y}`,
      stroke: `hsl(${rd.hue} 50% ${L - 8}%)`,
      strokeWidth: "1.6",
      fill: "none",
      opacity: "0.6"
    }), /*#__PURE__*/React.createElement("path", {
      d: `M ${rd.x + lean * 0.08 + sway * 0.5} ${rd.base - rd.h * 0.22} q ${(lean > 0 ? 1 : -1) * 34} ${-8} ${(lean > 0 ? 1 : -1) * 58} ${10}`,
      stroke: `hsl(${rd.hue} 40% ${L + 2}%)`,
      strokeWidth: Math.max(3, w * 0.34),
      strokeLinecap: "round",
      fill: "none"
    }));
  }));
}
function Silt({
  ph,
  group
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      left: -600,
      top: -200,
      width: 3100,
      height: 1500
    }
  }, SILT.filter(s => s.k === group).map((s, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      position: 'absolute',
      left: s.x + 600,
      top: s.y + 200,
      width: s.r * 2,
      height: s.r * 2,
      borderRadius: '50%',
      background: `rgba(214,248,240,${s.o})`,
      transform: `translate(${Math.sin(ph * 0.7 + s.ph * 6.28) * s.amp}px, ${Math.cos(ph * 0.9 + s.ph * 6.28) * s.amp * 0.6}px)`
    }
  })));
}
function Fish({
  w
}) {
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "-800 0 3600 1080",
    style: {
      position: 'absolute',
      left: -800,
      top: 0,
      width: 3600,
      height: 1080,
      overflow: 'visible'
    }
  }, FISH.map((f, i) => {
    const prog = (w * (0.6 + f.s * 0.5) + f.ph) % 1;
    const x = -700 + prog * 3300;
    const y = f.y + Math.sin(prog * 6.28 * 3) * f.bob;
    return /*#__PURE__*/React.createElement("g", {
      key: i,
      transform: `translate(${f.dir > 0 ? x : 2600 - x} ${y}) scale(${f.dir * f.s} ${f.s})`,
      opacity: f.o
    }, /*#__PURE__*/React.createElement("ellipse", {
      cx: "0",
      cy: "0",
      rx: "34",
      ry: "12",
      fill: "#0f3b46"
    }), /*#__PURE__*/React.createElement("path", {
      d: "M -32 0 L -56 -14 L -50 0 L -56 14 Z",
      fill: "#0f3b46"
    }));
  }));
}

/* axolotl ---------------------------------------------------------------- */
function GillBranch({
  side,
  index,
  x,
  y,
  ex,
  flutter,
  fphase = 0,
  lift,
  scale = 2,
  artic = 1
}) {
  const dir = side === 'L' ? -1 : 1;
  const spread = [-68, -26, 16][index];
  const ph = index * 0.7;
  const wave = k => Math.sin(fphase - ph - k * 0.9);
  const ang = spread - lift * 1.5 + wave(0) * (6 + index * 2);
  const len = 150 + index * 12;
  const tip = ex ? '#f7a9e2' : '#e78ccd';
  const core = ex ? '#8a55b0' : '#6b3f90';
  const curl = (index - 1) * 7;
  const halfW = u => 23 * (1 - u * 0.88);

  /* one continuous stem: a smoothly varying bend integrated along its length,
     so the stalk curves as a single piece with no visible joints */
  const N = 16;
  const pts = [];
  let px = 0,
    py = 0,
    a = 0;
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    pts.push({
      x: px,
      y: py,
      a,
      u
    });
    const bend = (curl * 0.9 + wave(1 + u * 2.4) * (11 + u * 16) * artic - lift * (0.5 + u * 0.7) * artic) / N;
    a += bend * Math.PI / 180;
    px += Math.cos(a) * (len / N);
    py += Math.sin(a) * (len / N);
  }
  const off = (p, sgn) => {
    const w = halfW(p.u) * sgn;
    return (p.x - Math.sin(p.a) * w).toFixed(1) + ' ' + (p.y + Math.cos(p.a) * w).toFixed(1);
  };
  const stem = 'M ' + pts.map(p => off(p, -1)).join(' L ') + ' L ' + pts.slice().reverse().map(p => off(p, 1)).join(' L ') + ' Z';
  const spine = 'M ' + pts.map(p => p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join(' L ');
  const fil = [0.16, 0.3, 0.44, 0.58, 0.72, 0.86];
  return /*#__PURE__*/React.createElement("g", {
    transform: `translate(${x} ${y}) scale(${dir * scale} ${scale}) rotate(${ang})`
  }, /*#__PURE__*/React.createElement("ellipse", {
    cx: "-3",
    cy: "0",
    rx: halfW(0) * 1.25,
    ry: halfW(0) * 1.05,
    fill: core
  }), /*#__PURE__*/React.createElement("path", {
    d: stem,
    fill: core
  }), fil.map((u, j) => {
    const p = pts[Math.round(u * N)];
    const deg = p.a * 180 / Math.PI;
    const fl = 48 - u * 18,
      fw = 12.5 - u * 5;
    return [1, -1].map(sgn => /*#__PURE__*/React.createElement("g", {
      key: `${j}${sgn}`,
      transform: `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)}) rotate(${deg + sgn * (66 - u * 32) + wave(1 + u * 2.4) * 9 * sgn * artic})`
    }, /*#__PURE__*/React.createElement("path", {
      d: `M 0 0 Q ${fl * 0.42} ${-fw} ${fl} 0 Q ${fl * 0.42} ${fw} 0 0 Z`,
      fill: tip,
      stroke: "#8e3b7e",
      strokeWidth: "2",
      strokeOpacity: "0.45"
    })));
  }), /*#__PURE__*/React.createElement("path", {
    d: spine,
    stroke: "#b98fd8",
    strokeWidth: "2.5",
    strokeLinecap: "round",
    fill: "none",
    opacity: "0.4"
  }));
}
function Eye({
  cx,
  e,
  blink,
  id
}) {
  const lid = clamp(Math.max(e.lid, blink), -0.08, 1);
  const rx = 42 * e.eyeSq,
    ry = 45 * e.eyeSq;
  return /*#__PURE__*/React.createElement("g", {
    transform: `translate(${cx} 222)`
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("clipPath", {
    id: `eyeClip${id}`
  }, /*#__PURE__*/React.createElement("ellipse", {
    cx: "0",
    cy: "0",
    rx: rx,
    ry: ry
  }))), /*#__PURE__*/React.createElement("ellipse", {
    cx: "0",
    cy: "0",
    rx: rx,
    ry: ry,
    fill: "#f6efff"
  }), /*#__PURE__*/React.createElement("g", {
    clipPath: `url(#eyeClip${id})`
  }, /*#__PURE__*/React.createElement("g", {
    transform: `translate(${e.eyeX * 0.8} ${e.eyeY * 0.8 + lid * 8})`
  }, /*#__PURE__*/React.createElement("ellipse", {
    cx: "0",
    cy: "0",
    rx: "26",
    ry: "28",
    fill: "#23102f"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "-9",
    cy: "-11",
    r: "8.5",
    fill: "#ffffff",
    opacity: "0.92"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "8",
    cy: "11",
    r: "4",
    fill: "#ffffff",
    opacity: "0.4"
  })), /*#__PURE__*/React.createElement("g", {
    transform: `translate(0 ${-ry - 2 + lid * (ry * 2 + 6)})`
  }, /*#__PURE__*/React.createElement("rect", {
    x: -rx - 6,
    y: -ry * 2 - 10,
    width: rx * 2 + 12,
    height: ry * 2 + 10,
    fill: "#5d3780"
  }), /*#__PURE__*/React.createElement("path", {
    d: `M ${-rx - 6} 0 h ${rx * 2 + 12}`,
    stroke: "#33194a",
    strokeWidth: "6",
    opacity: "0.9"
  }))), /*#__PURE__*/React.createElement("ellipse", {
    cx: "0",
    cy: "0",
    rx: rx,
    ry: ry,
    fill: "none",
    stroke: "#2e1642",
    strokeWidth: "4",
    opacity: "0.55"
  }));
}
function Axolotl({
  T,
  e,
  w,
  total,
  blinks,
  gillScale = 2,
  gillArtic = 1
}) {
  const fphase = 2 * Math.PI * 6 * (T / total);
  const flutter = Math.sin(fphase);
  const breath = Math.sin(2 * Math.PI * 8 * (T / total)) * 3;
  const blink = (() => {
    const times = blinks || [];
    let v = 0;
    for (const bt of times) {
      const d = T - bt;
      if (d > -0.06 && d < 0.24) v = Math.max(v, 1 - Math.abs(d - 0.09) / 0.15);
    }
    return clamp(v, 0, 1);
  })();
  const ex = e.spark > 0.4;
  const mouthTop = 338 + e.eyeY * 0.2;
  const cw = e.curve;
  return /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 600 560",
    style: {
      position: 'absolute',
      left: 990,
      top: 430,
      width: 300,
      height: 280,
      overflow: 'visible',
      transform: `translateY(${e.bob + breath}px) rotate(${e.tilt}deg)`,
      transformOrigin: '50% 70%'
    }
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("radialGradient", {
    id: "axoHead",
    cx: "42%",
    cy: "30%",
    r: "78%"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#7b4a99"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "52%",
    stopColor: "#5a3178"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "100%",
    stopColor: "#33184a"
  })), /*#__PURE__*/React.createElement("radialGradient", {
    id: "axoGlow",
    cx: "50%",
    cy: "50%",
    r: "50%"
  }, /*#__PURE__*/React.createElement("stop", {
    offset: "0%",
    stopColor: "#c79bff",
    stopOpacity: "0.5"
  }), /*#__PURE__*/React.createElement("stop", {
    offset: "100%",
    stopColor: "#c79bff",
    stopOpacity: "0"
  }))), /*#__PURE__*/React.createElement("ellipse", {
    cx: "300",
    cy: "290",
    rx: "330",
    ry: "270",
    fill: "url(#axoGlow)",
    opacity: "0.5"
  }), /*#__PURE__*/React.createElement("g", {
    opacity: "0.95"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M 470 488 L 566 512 L 528 442 Z",
    fill: "#4a2a66",
    opacity: "0.5"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 464 482 L 550 504 L 518 440 Z",
    fill: "#4a2a66"
  }), /*#__PURE__*/React.createElement("ellipse", {
    cx: "420",
    cy: "452",
    rx: "60",
    ry: "42",
    fill: "#3d2057",
    transform: "rotate(14 420 452)"
  }), /*#__PURE__*/React.createElement("ellipse", {
    cx: "316",
    cy: "406",
    rx: "118",
    ry: "72",
    fill: "#3d2057",
    transform: "rotate(6 316 406)"
  })), [0, 1, 2].map(i => /*#__PURE__*/React.createElement(GillBranch, {
    key: `l${i}`,
    side: "L",
    index: i,
    x: [157, 78, 48][i],
    y: [106, 164, 222][i],
    ex: ex,
    flutter: flutter,
    fphase: fphase,
    lift: e.gill,
    scale: gillScale,
    artic: gillArtic
  })), [0, 1, 2].map(i => /*#__PURE__*/React.createElement(GillBranch, {
    key: `r${i}`,
    side: "R",
    index: i,
    x: [443, 522, 552][i],
    y: [106, 164, 222][i],
    ex: ex,
    flutter: flutter,
    fphase: fphase,
    lift: e.gill,
    scale: gillScale,
    artic: gillArtic
  })), /*#__PURE__*/React.createElement("path", {
    d: "M 44 252 C 44 140 158 76 300 76 C 442 76 556 140 556 252 C 556 372 448 438 300 438 C 152 438 44 372 44 252 Z",
    fill: "url(#axoHead)"
  }), /*#__PURE__*/React.createElement("path", {
    d: "M 118 322 C 172 410 234 442 300 442 C 366 442 428 410 482 322 C 428 386 366 408 300 408 C 234 408 172 386 118 322 Z",
    fill: "#6d3f8d",
    opacity: "0.5"
  }), /*#__PURE__*/React.createElement("ellipse", {
    cx: "200",
    cy: "146",
    rx: "82",
    ry: "38",
    fill: "#ffffff",
    opacity: "0.09"
  }), /*#__PURE__*/React.createElement("ellipse", {
    cx: "168",
    cy: "306",
    rx: "44",
    ry: "22",
    fill: "#e07fc0",
    opacity: 0.1 + e.blush * 0.45
  }), /*#__PURE__*/React.createElement("ellipse", {
    cx: "432",
    cy: "306",
    rx: "44",
    ry: "22",
    fill: "#e07fc0",
    opacity: 0.1 + e.blush * 0.45
  }), /*#__PURE__*/React.createElement(Eye, {
    cx: 196,
    e: e,
    blink: blink,
    id: "L"
  }), /*#__PURE__*/React.createElement(Eye, {
    cx: 404,
    e: e,
    blink: blink,
    id: "R"
  }), /*#__PURE__*/React.createElement("g", {
    transform: `translate(196 ${146 + e.browL}) rotate(${e.rotL})`
  }, /*#__PURE__*/React.createElement("path", {
    d: "M -38 4 Q 0 -8 38 2",
    stroke: "#2a1340",
    strokeWidth: "13",
    strokeLinecap: "round",
    fill: "none",
    opacity: "0.8"
  })), /*#__PURE__*/React.createElement("g", {
    transform: `translate(404 ${146 + e.browR}) rotate(${e.rotR})`
  }, /*#__PURE__*/React.createElement("path", {
    d: "M -38 2 Q 0 -8 38 4",
    stroke: "#2a1340",
    strokeWidth: "13",
    strokeLinecap: "round",
    fill: "none",
    opacity: "0.8"
  })), /*#__PURE__*/React.createElement("circle", {
    cx: "272",
    cy: "300",
    r: "5.5",
    fill: "#2a1340",
    opacity: "0.45"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "328",
    cy: "300",
    r: "5.5",
    fill: "#2a1340",
    opacity: "0.45"
  }), /*#__PURE__*/React.createElement("g", {
    transform: `translate(300 ${mouthTop}) rotate(${e.mrot}) scale(${e.mw} 1)`
  }, e.open > 1 && /*#__PURE__*/React.createElement("path", {
    d: `M -88 -2 Q 0 ${cw * 1.2 + e.open * 1.6} 88 -2 Z`,
    fill: "#2a1038"
  }), /*#__PURE__*/React.createElement("path", {
    d: `M -98 0 Q 0 ${cw * 1.6} 98 0`,
    stroke: "#2a1340",
    strokeWidth: "9",
    strokeLinecap: "round",
    fill: "none"
  })), e.spark > 0.05 && /*#__PURE__*/React.createElement("g", {
    opacity: e.spark
  }, [[80, 90, 26], [520, 110, 20], [300, -6, 22]].map(([sx, sy, r], i) => /*#__PURE__*/React.createElement("path", {
    key: i,
    d: `M ${sx} ${sy - r} L ${sx + r * 0.3} ${sy - r * 0.3} L ${sx + r} ${sy} L ${sx + r * 0.3} ${sy + r * 0.3} L ${sx} ${sy + r} L ${sx - r * 0.3} ${sy + r * 0.3} L ${sx - r} ${sy} L ${sx - r * 0.3} ${sy - r * 0.3} Z`,
    fill: "#ffe9a8",
    transform: `rotate(${flutter * 10 + i * 12} ${sx} ${sy})`
  }))), e.zzz > 0.05 && /*#__PURE__*/React.createElement("g", {
    opacity: e.zzz
  }, [0, 1, 2].map(i => {
    const p = (w * 4 + i * 0.33) % 1;
    return /*#__PURE__*/React.createElement("text", {
      key: i,
      x: 540 + p * 60,
      y: 140 - p * 150,
      fontFamily: "Georgia, serif",
      fontSize: 30 + i * 8,
      fill: "#dfeffa",
      opacity: (1 - p) * 0.85
    }, "z");
  })));
}

/* scene shells (shared by the icon page) --------------------------------- */
function RiverBackdrop({
  ph,
  w,
  cam,
  glow = 1,
  particles = 1
}) {
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Layer, {
    depth: 0.14,
    cam: cam,
    blurBase: 2.5,
    z: 1
  }, /*#__PURE__*/React.createElement(WaterBody, {
    ph: ph
  })), /*#__PURE__*/React.createElement(Layer, {
    depth: 0.2,
    cam: cam,
    blurBase: 3,
    z: 2
  }, /*#__PURE__*/React.createElement(Surface, {
    ph: ph
  }), /*#__PURE__*/React.createElement(Fish, {
    w: w
  })), /*#__PURE__*/React.createElement(Layer, {
    depth: 0.3,
    cam: cam,
    blurBase: 1.5,
    z: 3
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: glow
    }
  }, /*#__PURE__*/React.createElement(Rays, {
    ph: ph
  }))), /*#__PURE__*/React.createElement(Layer, {
    depth: 0.42,
    cam: cam,
    blurBase: 1.5,
    z: 4
  }, /*#__PURE__*/React.createElement(Rocks, {
    ph: ph
  }), /*#__PURE__*/React.createElement(Reeds, {
    ph: ph,
    near: false
  })), /*#__PURE__*/React.createElement(Layer, {
    depth: 0.6,
    cam: cam,
    blurBase: 1,
    z: 5
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: particles
    }
  }, /*#__PURE__*/React.createElement(Silt, {
    ph: ph,
    group: 1
  }))), /*#__PURE__*/React.createElement(Layer, {
    depth: 1.5,
    cam: cam,
    blurBase: 6,
    z: 5.5
  }, /*#__PURE__*/React.createElement(Reeds, {
    ph: ph,
    near: true
  })));
}
function RiverForeground({
  ph,
  cam,
  glow = 1,
  particles = 1
}) {
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(Layer, {
    depth: 1.15,
    cam: cam,
    blurBase: 0,
    z: 7
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: particles
    }
  }, /*#__PURE__*/React.createElement(Silt, {
    ph: ph,
    group: 2
  }))), /*#__PURE__*/React.createElement(Layer, {
    depth: 1.5,
    cam: cam,
    blurBase: 6,
    z: 8
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      opacity: particles * 0.9
    }
  }, /*#__PURE__*/React.createElement(Silt, {
    ph: ph,
    group: 3
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      zIndex: 9,
      pointerEvents: 'none',
      background: 'radial-gradient(78% 70% at 50% 42%, rgba(0,0,0,0) 42%, rgba(3,20,28,0.62) 100%)'
    }
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      zIndex: 10,
      pointerEvents: 'none',
      background: `linear-gradient(rgba(96,200,196,${0.1 * glow}) 0%, rgba(10,60,72,0.06) 50%, rgba(4,24,32,0.3) 100%)`
    }
  }));
}

/* canonical full-composition timing — clip pages sample windows of this ---- */
const CANON = [['River', 4.2], ['Approach', 2.8], ['Attentive', 2.2], ['Idle1', 1.4], ['Thinking', 2.6], ['Idle2', 2], ['Idea', 2.2], ['Idle3', 2], ['Confused', 2.4], ['Idle4', 2], ['Focused', 2.4], ['Idle5', 2], ['Sleepy', 3], ['Idle6', 2], ['Pullback', 3.6]];
const CANON_CUES = {};
const CANON_DUR = {};
let _acc = 0;
CANON.forEach(([n, d]) => {
  CANON_CUES[n] = _acc;
  CANON_DUR[n] = d;
  _acc += d;
});
const CANON_TOTAL = _acc;

/* OM_CLIP: [[firstScene, lastScene], …] — segments played back to back */
function clipSegments(clip) {
  return clip.map(([a, b]) => [CANON_CUES[a], CANON_CUES[b] + CANON_DUR[b]]);
}
function clipMap(clip, localT) {
  const segs = clipSegments(clip);
  const len = segs.reduce((s, [a, b]) => s + (b - a), 0);
  let x = (localT % len + len) % len;
  for (const [a, b] of segs) {
    if (x <= b - a) return {
      g: a + x,
      len
    };
    x -= b - a;
  }
  return {
    g: segs[segs.length - 1][1],
    len
  };
}

/* piece ------------------------------------------------------------------ */
function AxolotlRiver({
  t = {}
}) {
  const comp = useComposition();
  const clip = window.OM_CLIP;
  let T = comp.T,
    CUES = comp.CUES,
    total = comp.authoredTotal || 31.4;
  let w = T / total % 1;
  if (clip) {
    const m = clipMap(clip, comp.T);
    w = comp.T / m.len % 1;
    T = m.g;
    CUES = CANON_CUES;
    total = CANON_TOTAL;
  }
  const ph = 2 * Math.PI * 2 * w;
  const cam = cameraAt(T, CUES, total, t.viewFit ?? 0.44);
  const e = exprAt(T, CUES, total);
  const blinks = [1, 2, 3, 4, 5, 6].map(n => CUES['Idle' + n] + 0.42).filter(v => !isNaN(v));
  const glow = t.glow == null ? 1 : t.glow;
  const particles = t.particles == null ? 1 : t.particles;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0,
      overflow: 'hidden',
      background: '#05202a'
    }
  }, /*#__PURE__*/React.createElement(RiverBackdrop, {
    ph: ph,
    w: w,
    cam: cam,
    glow: glow,
    particles: particles
  }), /*#__PURE__*/React.createElement(Layer, {
    depth: 1,
    cam: cam,
    z: 6
  }, /*#__PURE__*/React.createElement(Axolotl, {
    T: T,
    e: e,
    w: w,
    total: total,
    blinks: blinks,
    gillScale: t.gillScale ?? 2,
    gillArtic: t.gillArtic ?? 1
  })), /*#__PURE__*/React.createElement(RiverForeground, {
    ph: ph,
    cam: cam,
    glow: glow,
    particles: particles
  }));
}
window.AxolotlRiver = AxolotlRiver;
window.AxolotlHead = Axolotl;
window.AXO_REST = REST;
window.AXO_SCENE = {
  Layer,
  RiverBackdrop,
  RiverForeground,
  Axolotl,
  W,
  H
};

})();
;(function(){ /* ── tweaks-panel.jsx ── */
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
// Copied omelette starter. Re-running copy_starter_component with this kind overwrites this file with the latest version (page content is unaffected).

/* BEGIN USAGE */
// tweaks-panel.jsx
// Reusable Tweaks shell + form-control helpers.
// Exports (to window): useTweaks, TweaksPanel, TweakSection, TweakRow, TweakSlider,
//   TweakToggle, TweakRadio, TweakSelect, TweakText, TweakNumber, TweakColor, TweakButton.
//
// Owns the host protocol (listens for __activate_edit_mode / __deactivate_edit_mode,
// posts __edit_mode_available / __edit_mode_set_keys / __edit_mode_dismissed) so
// individual prototypes don't re-roll it. Ships a consistent set of controls so you
// don't hand-draw <input type="range">, segmented radios, steppers, etc.
//
// Usage (in an HTML file that loads React + Babel):
//
//   const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
//     "primaryColor": "#D97757",
//     "palette": ["#D97757", "#29261b", "#f6f4ef"],
//     "fontSize": 16,
//     "density": "regular",
//     "dark": false
//   }/*EDITMODE-END*/;
//
//   function App() {
//     const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
//     return (
//       <div style={{ fontSize: t.fontSize, color: t.primaryColor }}>
//         Hello
//         <TweaksPanel>
//           <TweakSection label="Typography" />
//           <TweakSlider label="Font size" value={t.fontSize} min={10} max={32} unit="px"
//                        onChange={(v) => setTweak('fontSize', v)} />
//           <TweakRadio  label="Density" value={t.density}
//                        options={['compact', 'regular', 'comfy']}
//                        onChange={(v) => setTweak('density', v)} />
//           <TweakSection label="Theme" />
//           <TweakColor  label="Primary" value={t.primaryColor}
//                        options={['#D97757', '#2A6FDB', '#1F8A5B', '#7A5AE0']}
//                        onChange={(v) => setTweak('primaryColor', v)} />
//           <TweakColor  label="Palette" value={t.palette}
//                        options={[['#D97757', '#29261b', '#f6f4ef'],
//                                  ['#475569', '#0f172a', '#f1f5f9']]}
//                        onChange={(v) => setTweak('palette', v)} />
//           <TweakToggle label="Dark mode" value={t.dark}
//                        onChange={(v) => setTweak('dark', v)} />
//         </TweaksPanel>
//       </div>
//     );
//   }
//
// TweakRadio is the segmented control for 2–3 short options (auto-falls-back to
// TweakSelect past ~16/~10 chars per label); reach for TweakSelect directly when
// options are many or long. For color tweaks always curate 3-4 options rather than
// a free picker; an option can also be a whole 2–5 color palette (the stored value
// is the array). The Tweak* controls are a floor, not a ceiling — build custom
// controls inside the panel if a tweak calls for UI they don't cover.
/* END USAGE */
// ─────────────────────────────────────────────────────────────────────────────

const __TWEAKS_STYLE = `
  .twk-panel{position:fixed;right:16px;bottom:16px;z-index:2147483646;width:280px;
    max-height:calc(100vh - 32px);display:flex;flex-direction:column;
    transform:scale(var(--dc-inv-zoom,1));transform-origin:bottom right;
    background:rgba(250,249,247,.78);color:#29261b;
    -webkit-backdrop-filter:blur(24px) saturate(160%);backdrop-filter:blur(24px) saturate(160%);
    border:.5px solid rgba(255,255,255,.6);border-radius:14px;
    box-shadow:0 1px 0 rgba(255,255,255,.5) inset,0 12px 40px rgba(0,0,0,.18);
    font:11.5px/1.4 ui-sans-serif,system-ui,-apple-system,sans-serif;overflow:hidden}
  .twk-hd{display:flex;align-items:center;justify-content:space-between;
    padding:10px 8px 10px 14px;cursor:move;user-select:none}
  .twk-hd b{font-size:12px;font-weight:600;letter-spacing:.01em}
  .twk-x{appearance:none;border:0;background:transparent;color:rgba(41,38,27,.55);
    width:22px;height:22px;border-radius:6px;cursor:default;font-size:13px;line-height:1}
  .twk-x:hover{background:rgba(0,0,0,.06);color:#29261b}
  .twk-body{padding:2px 14px 14px;display:flex;flex-direction:column;gap:10px;
    overflow-y:auto;overflow-x:hidden;min-height:0;
    scrollbar-width:thin;scrollbar-color:rgba(0,0,0,.15) transparent}
  .twk-body::-webkit-scrollbar{width:8px}
  .twk-body::-webkit-scrollbar-track{background:transparent;margin:2px}
  .twk-body::-webkit-scrollbar-thumb{background:rgba(0,0,0,.15);border-radius:4px;
    border:2px solid transparent;background-clip:content-box}
  .twk-body::-webkit-scrollbar-thumb:hover{background:rgba(0,0,0,.25);
    border:2px solid transparent;background-clip:content-box}
  .twk-row{display:flex;flex-direction:column;gap:5px}
  .twk-row-h{flex-direction:row;align-items:center;justify-content:space-between;gap:10px}
  .twk-lbl{display:flex;justify-content:space-between;align-items:baseline;
    color:rgba(41,38,27,.72)}
  .twk-lbl>span:first-child{font-weight:500}
  .twk-val{color:rgba(41,38,27,.5);font-variant-numeric:tabular-nums}

  .twk-sect{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
    color:rgba(41,38,27,.45);padding:10px 0 0}
  .twk-sect:first-child{padding-top:0}

  .twk-field{appearance:none;box-sizing:border-box;width:100%;min-width:0;height:26px;padding:0 8px;
    border:.5px solid rgba(0,0,0,.1);border-radius:7px;
    background:rgba(255,255,255,.6);color:inherit;font:inherit;outline:none}
  .twk-field:focus{border-color:rgba(0,0,0,.25);background:rgba(255,255,255,.85)}
  select.twk-field{padding-right:22px;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='rgba(0,0,0,.5)' d='M0 0h10L5 6z'/></svg>");
    background-repeat:no-repeat;background-position:right 8px center}

  .twk-slider{appearance:none;-webkit-appearance:none;width:100%;height:4px;margin:6px 0;
    border-radius:999px;background:rgba(0,0,0,.12);outline:none}
  .twk-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
    width:14px;height:14px;border-radius:50%;background:#fff;
    border:.5px solid rgba(0,0,0,.12);box-shadow:0 1px 3px rgba(0,0,0,.2);cursor:default}
  .twk-slider::-moz-range-thumb{width:14px;height:14px;border-radius:50%;
    background:#fff;border:.5px solid rgba(0,0,0,.12);box-shadow:0 1px 3px rgba(0,0,0,.2);cursor:default}

  .twk-seg{position:relative;display:flex;padding:2px;border-radius:8px;
    background:rgba(0,0,0,.06);user-select:none}
  .twk-seg-thumb{position:absolute;top:2px;bottom:2px;border-radius:6px;
    background:rgba(255,255,255,.9);box-shadow:0 1px 2px rgba(0,0,0,.12);
    transition:left .15s cubic-bezier(.3,.7,.4,1),width .15s}
  .twk-seg.dragging .twk-seg-thumb{transition:none}
  .twk-seg button{appearance:none;position:relative;z-index:1;flex:1;border:0;
    background:transparent;color:inherit;font:inherit;font-weight:500;min-height:22px;
    border-radius:6px;cursor:default;padding:4px 6px;line-height:1.2;
    overflow-wrap:anywhere}

  .twk-toggle{position:relative;width:32px;height:18px;border:0;border-radius:999px;
    background:rgba(0,0,0,.15);transition:background .15s;cursor:default;padding:0}
  .twk-toggle[data-on="1"]{background:#34c759}
  .twk-toggle i{position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;
    background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .15s}
  .twk-toggle[data-on="1"] i{transform:translateX(14px)}

  .twk-num{display:flex;align-items:center;box-sizing:border-box;min-width:0;height:26px;padding:0 0 0 8px;
    border:.5px solid rgba(0,0,0,.1);border-radius:7px;background:rgba(255,255,255,.6)}
  .twk-num-lbl{font-weight:500;color:rgba(41,38,27,.6);cursor:ew-resize;
    user-select:none;padding-right:8px}
  .twk-num input{flex:1;min-width:0;height:100%;border:0;background:transparent;
    font:inherit;font-variant-numeric:tabular-nums;text-align:right;padding:0 8px 0 0;
    outline:none;color:inherit;-moz-appearance:textfield}
  .twk-num input::-webkit-inner-spin-button,.twk-num input::-webkit-outer-spin-button{
    -webkit-appearance:none;margin:0}
  .twk-num-unit{padding-right:8px;color:rgba(41,38,27,.45)}

  .twk-btn{appearance:none;height:26px;padding:0 12px;border:0;border-radius:7px;
    background:rgba(0,0,0,.78);color:#fff;font:inherit;font-weight:500;cursor:default}
  .twk-btn:hover{background:rgba(0,0,0,.88)}
  .twk-btn.secondary{background:rgba(0,0,0,.06);color:inherit}
  .twk-btn.secondary:hover{background:rgba(0,0,0,.1)}

  .twk-swatch{appearance:none;-webkit-appearance:none;width:56px;height:22px;
    border:.5px solid rgba(0,0,0,.1);border-radius:6px;padding:0;cursor:default;
    background:transparent;flex-shrink:0}
  .twk-swatch::-webkit-color-swatch-wrapper{padding:0}
  .twk-swatch::-webkit-color-swatch{border:0;border-radius:5.5px}
  .twk-swatch::-moz-color-swatch{border:0;border-radius:5.5px}

  .twk-chips{display:flex;gap:6px}
  .twk-chip{position:relative;appearance:none;flex:1;min-width:0;height:46px;
    padding:0;border:0;border-radius:6px;overflow:hidden;cursor:default;
    box-shadow:0 0 0 .5px rgba(0,0,0,.12),0 1px 2px rgba(0,0,0,.06);
    transition:transform .12s cubic-bezier(.3,.7,.4,1),box-shadow .12s}
  .twk-chip:hover{transform:translateY(-1px);
    box-shadow:0 0 0 .5px rgba(0,0,0,.18),0 4px 10px rgba(0,0,0,.12)}
  .twk-chip[data-on="1"]{box-shadow:0 0 0 1.5px rgba(0,0,0,.85),
    0 2px 6px rgba(0,0,0,.15)}
  .twk-chip>span{position:absolute;top:0;bottom:0;right:0;width:34%;
    display:flex;flex-direction:column;box-shadow:-1px 0 0 rgba(0,0,0,.1)}
  .twk-chip>span>i{flex:1;box-shadow:0 -1px 0 rgba(0,0,0,.1)}
  .twk-chip>span>i:first-child{box-shadow:none}
  .twk-chip svg{position:absolute;top:6px;left:6px;width:13px;height:13px;
    filter:drop-shadow(0 1px 1px rgba(0,0,0,.3))}
`;

// ── useTweaks ───────────────────────────────────────────────────────────────
// Single source of truth for tweak values. setTweak persists via the host
// (__edit_mode_set_keys → host rewrites the EDITMODE block on disk).
function useTweaks(defaults) {
  const [values, setValues] = React.useState(defaults);
  // Accepts either setTweak('key', value) or setTweak({ key: value, ... }) so a
  // useState-style call doesn't write a "[object Object]" key into the persisted
  // JSON block.
  const setTweak = React.useCallback((keyOrEdits, val) => {
    const edits = typeof keyOrEdits === 'object' && keyOrEdits !== null ? keyOrEdits : {
      [keyOrEdits]: val
    };
    setValues(prev => ({
      ...prev,
      ...edits
    }));
    window.parent.postMessage({
      type: '__edit_mode_set_keys',
      edits
    }, '*');
    // Same-window signal so in-page listeners (deck-stage rail thumbnails)
    // can react — the parent message only reaches the host, not peers.
    window.dispatchEvent(new CustomEvent('tweakchange', {
      detail: edits
    }));
  }, []);
  return [values, setTweak];
}

// ── TweaksPanel ─────────────────────────────────────────────────────────────
// Floating shell. Registers the protocol listener BEFORE announcing
// availability — if the announce ran first, the host's activate could land
// before our handler exists and the toolbar toggle would silently no-op.
// The close button posts __edit_mode_dismissed so the host's toolbar toggle
// flips off in lockstep; the host echoes __deactivate_edit_mode back which
// is what actually hides the panel.
function TweaksPanel({
  title = 'Tweaks',
  children
}) {
  const [open, setOpen] = React.useState(false);
  const dragRef = React.useRef(null);
  const offsetRef = React.useRef({
    x: 16,
    y: 16
  });
  const PAD = 16;
  const clampToViewport = React.useCallback(() => {
    const panel = dragRef.current;
    if (!panel) return;
    const w = panel.offsetWidth,
      h = panel.offsetHeight;
    const maxRight = Math.max(PAD, window.innerWidth - w - PAD);
    const maxBottom = Math.max(PAD, window.innerHeight - h - PAD);
    offsetRef.current = {
      x: Math.min(maxRight, Math.max(PAD, offsetRef.current.x)),
      y: Math.min(maxBottom, Math.max(PAD, offsetRef.current.y))
    };
    panel.style.right = offsetRef.current.x + 'px';
    panel.style.bottom = offsetRef.current.y + 'px';
  }, []);
  React.useEffect(() => {
    if (!open) return;
    clampToViewport();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', clampToViewport);
      return () => window.removeEventListener('resize', clampToViewport);
    }
    const ro = new ResizeObserver(clampToViewport);
    ro.observe(document.documentElement);
    return () => ro.disconnect();
  }, [open, clampToViewport]);
  React.useEffect(() => {
    const onMsg = e => {
      const t = e?.data?.type;
      if (t === '__activate_edit_mode') setOpen(true);else if (t === '__deactivate_edit_mode') setOpen(false);
    };
    window.addEventListener('message', onMsg);
    window.parent.postMessage({
      type: '__edit_mode_available'
    }, '*');
    return () => window.removeEventListener('message', onMsg);
  }, []);
  const dismiss = () => {
    setOpen(false);
    window.parent.postMessage({
      type: '__edit_mode_dismissed'
    }, '*');
  };
  const onDragStart = e => {
    const panel = dragRef.current;
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    const sx = e.clientX,
      sy = e.clientY;
    const startRight = window.innerWidth - r.right;
    const startBottom = window.innerHeight - r.bottom;
    const move = ev => {
      offsetRef.current = {
        x: startRight - (ev.clientX - sx),
        y: startBottom - (ev.clientY - sy)
      };
      clampToViewport();
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  // data-om-starter: inert presence marker — Claude Design's starter-usage
  // probe reads it. The closed panel renders nothing, so the marker rides
  // the <html> element as an attribute instead of a rendered node — zero
  // elements added, so page CSS (even structural selectors like
  // :nth-child) can never observe it. It records that the page WIRES a
  // tweaks panel, whether or not the panel is open. Keep this effect.
  React.useEffect(() => {
    document.documentElement.setAttribute('data-om-starter', 'tweaks-panel');
    return () => document.documentElement.removeAttribute('data-om-starter');
  }, []);
  if (!open) return null;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("style", null, __TWEAKS_STYLE), /*#__PURE__*/React.createElement("div", {
    ref: dragRef,
    className: "twk-panel",
    "data-omelette-chrome": "",
    style: {
      right: offsetRef.current.x,
      bottom: offsetRef.current.y
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "twk-hd",
    onMouseDown: onDragStart
  }, /*#__PURE__*/React.createElement("b", null, title), /*#__PURE__*/React.createElement("button", {
    className: "twk-x",
    "aria-label": "Close tweaks",
    onMouseDown: e => e.stopPropagation(),
    onClick: dismiss
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "twk-body"
  }, children)));
}

// ── Layout helpers ──────────────────────────────────────────────────────────

function TweakSection({
  label,
  children
}) {
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "twk-sect"
  }, label), children);
}
function TweakRow({
  label,
  value,
  children,
  inline = false
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: inline ? 'twk-row twk-row-h' : 'twk-row'
  }, /*#__PURE__*/React.createElement("div", {
    className: "twk-lbl"
  }, /*#__PURE__*/React.createElement("span", null, label), value != null && /*#__PURE__*/React.createElement("span", {
    className: "twk-val"
  }, value)), children);
}

// ── Controls ────────────────────────────────────────────────────────────────

function TweakSlider({
  label,
  value,
  min = 0,
  max = 100,
  step = 1,
  unit = '',
  onChange
}) {
  return /*#__PURE__*/React.createElement(TweakRow, {
    label: label,
    value: `${value}${unit}`
  }, /*#__PURE__*/React.createElement("input", {
    type: "range",
    className: "twk-slider",
    min: min,
    max: max,
    step: step,
    value: value,
    onChange: e => onChange(Number(e.target.value))
  }));
}
function TweakToggle({
  label,
  value,
  onChange
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "twk-row twk-row-h"
  }, /*#__PURE__*/React.createElement("div", {
    className: "twk-lbl"
  }, /*#__PURE__*/React.createElement("span", null, label)), /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: "twk-toggle",
    "data-on": value ? '1' : '0',
    role: "switch",
    "aria-checked": !!value,
    onClick: () => onChange(!value)
  }, /*#__PURE__*/React.createElement("i", null)));
}
function TweakRadio({
  label,
  value,
  options,
  onChange
}) {
  const trackRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);
  // The active value is read by pointer-move handlers attached for the lifetime
  // of a drag — ref it so a stale closure doesn't fire onChange for every move.
  const valueRef = React.useRef(value);
  valueRef.current = value;

  // Segments wrap mid-word once per-segment width runs out. The track is
  // ~248px (280 panel − 28 body pad − 4 seg pad), each button loses 12px
  // to its own padding, and 11.5px system-ui averages ~6.3px/char — so 2
  // options fit ~16 chars each, 3 fit ~10. Past that (or >3 options), fall
  // back to a dropdown rather than wrap.
  const labelLen = o => String(typeof o === 'object' ? o.label : o).length;
  const maxLen = options.reduce((m, o) => Math.max(m, labelLen(o)), 0);
  const fitsAsSegments = maxLen <= ({
    2: 16,
    3: 10
  }[options.length] ?? 0);
  if (!fitsAsSegments) {
    // <select> emits strings — map back to the original option value so the
    // fallback stays type-preserving (numbers, booleans) like the segment path.
    const resolve = s => {
      const m = options.find(o => String(typeof o === 'object' ? o.value : o) === s);
      return m === undefined ? s : typeof m === 'object' ? m.value : m;
    };
    return /*#__PURE__*/React.createElement(TweakSelect, {
      label: label,
      value: value,
      options: options,
      onChange: s => onChange(resolve(s))
    });
  }
  const opts = options.map(o => typeof o === 'object' ? o : {
    value: o,
    label: o
  });
  const idx = Math.max(0, opts.findIndex(o => o.value === value));
  const n = opts.length;
  const segAt = clientX => {
    const r = trackRef.current.getBoundingClientRect();
    const inner = r.width - 4;
    const i = Math.floor((clientX - r.left - 2) / inner * n);
    return opts[Math.max(0, Math.min(n - 1, i))].value;
  };
  const onPointerDown = e => {
    setDragging(true);
    const v0 = segAt(e.clientX);
    if (v0 !== valueRef.current) onChange(v0);
    const move = ev => {
      if (!trackRef.current) return;
      const v = segAt(ev.clientX);
      if (v !== valueRef.current) onChange(v);
    };
    const up = () => {
      setDragging(false);
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  return /*#__PURE__*/React.createElement(TweakRow, {
    label: label
  }, /*#__PURE__*/React.createElement("div", {
    ref: trackRef,
    role: "radiogroup",
    onPointerDown: onPointerDown,
    className: dragging ? 'twk-seg dragging' : 'twk-seg'
  }, /*#__PURE__*/React.createElement("div", {
    className: "twk-seg-thumb",
    style: {
      left: `calc(2px + ${idx} * (100% - 4px) / ${n})`,
      width: `calc((100% - 4px) / ${n})`
    }
  }), opts.map(o => /*#__PURE__*/React.createElement("button", {
    key: o.value,
    type: "button",
    role: "radio",
    "aria-checked": o.value === value
  }, o.label))));
}
function TweakSelect({
  label,
  value,
  options,
  onChange
}) {
  return /*#__PURE__*/React.createElement(TweakRow, {
    label: label
  }, /*#__PURE__*/React.createElement("select", {
    className: "twk-field",
    value: value,
    onChange: e => onChange(e.target.value)
  }, options.map(o => {
    const v = typeof o === 'object' ? o.value : o;
    const l = typeof o === 'object' ? o.label : o;
    return /*#__PURE__*/React.createElement("option", {
      key: v,
      value: v
    }, l);
  })));
}
function TweakText({
  label,
  value,
  placeholder,
  onChange
}) {
  return /*#__PURE__*/React.createElement(TweakRow, {
    label: label
  }, /*#__PURE__*/React.createElement("input", {
    className: "twk-field",
    type: "text",
    value: value,
    placeholder: placeholder,
    onChange: e => onChange(e.target.value)
  }));
}
function TweakNumber({
  label,
  value,
  min,
  max,
  step = 1,
  unit = '',
  onChange
}) {
  const clamp = n => {
    if (min != null && n < min) return min;
    if (max != null && n > max) return max;
    return n;
  };
  const startRef = React.useRef({
    x: 0,
    val: 0
  });
  const onScrubStart = e => {
    e.preventDefault();
    startRef.current = {
      x: e.clientX,
      val: value
    };
    const decimals = (String(step).split('.')[1] || '').length;
    const move = ev => {
      const dx = ev.clientX - startRef.current.x;
      const raw = startRef.current.val + dx * step;
      const snapped = Math.round(raw / step) * step;
      onChange(clamp(Number(snapped.toFixed(decimals))));
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "twk-num"
  }, /*#__PURE__*/React.createElement("span", {
    className: "twk-num-lbl",
    onPointerDown: onScrubStart
  }, label), /*#__PURE__*/React.createElement("input", {
    type: "number",
    value: value,
    min: min,
    max: max,
    step: step,
    onChange: e => onChange(clamp(Number(e.target.value)))
  }), unit && /*#__PURE__*/React.createElement("span", {
    className: "twk-num-unit"
  }, unit));
}

// Relative-luminance contrast pick — checkmarks drawn over a swatch need to
// read on both #111 and #fafafa without per-option configuration. Hex input
// only (#rgb / #rrggbb); named or rgb()/hsl() colors fall through to "light".
function __twkIsLight(hex) {
  const h = String(hex).replace('#', '');
  const x = h.length === 3 ? h.replace(/./g, c => c + c) : h.padEnd(6, '0');
  const n = parseInt(x.slice(0, 6), 16);
  if (Number.isNaN(n)) return true;
  const r = n >> 16 & 255,
    g = n >> 8 & 255,
    b = n & 255;
  return r * 299 + g * 587 + b * 114 > 148000;
}
const __TwkCheck = ({
  light
}) => /*#__PURE__*/React.createElement("svg", {
  viewBox: "0 0 14 14",
  "aria-hidden": "true"
}, /*#__PURE__*/React.createElement("path", {
  d: "M3 7.2 5.8 10 11 4.2",
  fill: "none",
  strokeWidth: "2.2",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  stroke: light ? 'rgba(0,0,0,.78)' : '#fff'
}));

// TweakColor — curated color/palette picker. Each option is either a single
// hex string or an array of 1-5 hex strings; the card adapts — a lone color
// renders solid, a palette renders colors[0] as the hero (left ~2/3) with the
// rest stacked in a sharp column on the right. onChange emits the
// option in the shape it was passed (string stays string, array stays array).
// Without options it falls back to the native color input for back-compat.
function TweakColor({
  label,
  value,
  options,
  onChange
}) {
  if (!options || !options.length) {
    return /*#__PURE__*/React.createElement("div", {
      className: "twk-row twk-row-h"
    }, /*#__PURE__*/React.createElement("div", {
      className: "twk-lbl"
    }, /*#__PURE__*/React.createElement("span", null, label)), /*#__PURE__*/React.createElement("input", {
      type: "color",
      className: "twk-swatch",
      value: value,
      onChange: e => onChange(e.target.value)
    }));
  }
  // Native <input type=color> emits lowercase hex per the HTML spec, so
  // compare case-insensitively. String() guards JSON.stringify(undefined),
  // which returns the primitive undefined (no .toLowerCase).
  const key = o => String(JSON.stringify(o)).toLowerCase();
  const cur = key(value);
  return /*#__PURE__*/React.createElement(TweakRow, {
    label: label
  }, /*#__PURE__*/React.createElement("div", {
    className: "twk-chips",
    role: "radiogroup"
  }, options.map((o, i) => {
    const colors = Array.isArray(o) ? o : [o];
    const [hero, ...rest] = colors;
    const sup = rest.slice(0, 4);
    const on = key(o) === cur;
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      type: "button",
      className: "twk-chip",
      role: "radio",
      "aria-checked": on,
      "data-on": on ? '1' : '0',
      "aria-label": colors.join(', '),
      title: colors.join(' · '),
      style: {
        background: hero
      },
      onClick: () => onChange(o)
    }, sup.length > 0 && /*#__PURE__*/React.createElement("span", null, sup.map((c, j) => /*#__PURE__*/React.createElement("i", {
      key: j,
      style: {
        background: c
      }
    }))), on && /*#__PURE__*/React.createElement(__TwkCheck, {
      light: __twkIsLight(hero)
    }));
  })));
}
function TweakButton({
  label,
  onClick,
  secondary = false
}) {
  return /*#__PURE__*/React.createElement("button", {
    type: "button",
    className: secondary ? 'twk-btn secondary' : 'twk-btn',
    onClick: onClick
  }, label);
}
Object.assign(window, {
  useTweaks,
  TweaksPanel,
  TweakSection,
  TweakRow,
  TweakSlider,
  TweakToggle,
  TweakRadio,
  TweakSelect,
  TweakText,
  TweakNumber,
  TweakColor,
  TweakButton
});

})();
;(function(){ /* ── river-app.jsx ── */
const {
  CompositionStage,
  useTweaks,
  TweaksPanel,
  TweakSection,
  TweakToggle,
  TweakSlider
} = window;
function RiverApp(dcProps) {
  const [tw, setTweak] = useTweaks(window.TWEAK_DEFAULTS || {});
  const t = {
    ...tw,
    ...(dcProps || {})
  };
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: 0
    }
  }, /*#__PURE__*/React.createElement(CompositionStage, {
    width: 1080,
    height: 1080,
    bg: "#05202a",
    scenes: window.OM_SCENES,
    playback: window.OM_PLAYBACK
  }, /*#__PURE__*/React.createElement(window.AxolotlRiver, {
    t: t
  })), /*#__PURE__*/React.createElement(TweaksPanel, null, /*#__PURE__*/React.createElement(TweakSection, {
    label: "Scene"
  }), /*#__PURE__*/React.createElement(TweakSlider, {
    label: "Light shafts",
    value: t.glow,
    min: 0,
    max: 1.6,
    step: 0.05,
    onChange: v => setTweak('glow', v)
  }), /*#__PURE__*/React.createElement(TweakSlider, {
    label: "Drifting silt",
    value: t.particles,
    min: 0,
    max: 1.6,
    step: 0.05,
    onChange: v => setTweak('particles', v)
  }), /*#__PURE__*/React.createElement(TweakSection, {
    label: "Tools"
  }), /*#__PURE__*/React.createElement(TweakToggle, {
    label: "Motion editor",
    value: t.motionEditor,
    onChange: v => setTweak('motionEditor', v)
  })));
}
window.RiverApp = RiverApp;

})();
