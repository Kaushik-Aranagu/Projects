# Gesture policy

Kaushik, Oct 23: touches and swipes are not stable. There are accidental touches and accidental swipes in every room.
He wants gestures that are stable, consistent and easy. This page is the one policy. The numbers are code
(`TesseraCore/Sources/TesseraCore/Gesture/GestureTuning.swift`, pinned by `GestureTuningTests`), and the SwiftUI side
is `DesignSystem/GestureKit.swift`. A view never invents a distance, an axis rule or a velocity of its own.

## The rules

1. **A touch is a tap until it moves.** Under 10 pt of travel it is a tap (`tapSlop`). A drag that starts an action
   does not begin before 10 pt (`dragStart`, the SwiftUI `minimumDistance`). Never `minimumDistance: 0` on a body, a
   list or a row: it makes every touch a drag and starves the buttons under it. The only `0`s allowed are the
   instruments that must feel the finger land (the weight reel, reps, effort, the mosaic touch feed, Tess's soul).
2. **The axis is decided once.** At 12 pt of travel (`axisLock`) the drag is horizontal only if it leads the vertical
   drift by 1.5x (`horizontalBias`); otherwise it is vertical and a row swipe ignores it for the rest of the touch.
   Page turns and pulls need a 2x lead (`pagerBias`). A drag that locked as a scroll never becomes a swipe because the
   finger drifted sideways later, and a swipe never gives up because it drifted down.
3. **A commit is a decision, not an accident.** A row swipe arms at 96 pt (`armDistance`; 72 when the row does not
   follow the finger, `armUndrawn`), plays the house tick once (8 pt of hysteresis, so it never chatters) and commits on
   release while armed. Short of that, only a real flick commits: 500 pt/s in the swipe's direction after at least 28
   pt of travel. A release moving back against the pull at 350 pt/s cancels, even past the line. A page turn commits
   past 60 pt; a pull past 56 pt; both with the same flick rule. A 12 pt twitch at any speed commits nothing.
4. **The system back edge is not ours.** 0 to 24 pt on the leading edge belongs to the back gesture. An action that
   starts a rightward drag (Done, an action word, a page turn to the previous page) must start at least 28 pt in
   (`actionInset`); `SwipeTracker` and `GestureTuning.pageTurn` refuse it otherwise. Leftward drags are free. Top and
   bottom: actions start at least 24 pt in from the status bar and the home indicator.
5. **A tap after a wheel is the finger stopping it.** After a custom scroller (a wheel, a rail, a pager) lets go,
   taps count again 120 ms later (`TapHoldoff`, `settleHoldoff`). `ScrollView` already does this itself.
6. **Holds come in three sizes.** Quick 0.18 s (the hold-to-finish strike, the apps list pick-up), pick 0.35 s (start
   a reorder), open 0.5 s (open a detail or a sheet). Any hold is cancelled by 8 pt of movement (`holdSlop`). A
   long press that only adds a detail to a row is `simultaneousGesture(LongPressGesture)`; it never wraps a
   `DragGesture(minimumDistance: 0)` in a `highPriorityGesture` (that is what swallowed the apps list taps in
   release 77).
7. **Every commit on release can be undone.** The action goes through `ActionEngine`, so the one Undo (the op log's,
   10 s, `undoSeconds`) is on offer. A commit that cannot be undone needs a hold, not a swipe.
8. **One haptic vocabulary.** Arm plays `.selection` once, from `rowSwipe` itself, so a row does not add its own.
   The commit plays the action's own moment (`.done`, `.made`, `.notNow`, `.pinned`). Cancel plays nothing. A bound
   plays `.bound` once per push. All through `.tesseraFeedback`, never raw.
9. **Every swipe has a name for VoiceOver.** Each swipe is also an accessibility action ("Done", "Not now", "Remove").

## The three modifiers

```swift
.rowSwipe(.trailing, arm: 96, limit: 144) { event in   // .pulled, .armed, .committed(side), .cancelled, .ended
    switch event { ... }
}
.pageSwipe { next in ... }    // next is true for a drag to the left
.pullGesture { down in ... }  // a pull down or up that opens or closes something
```

`rowSwipe` draws `.pulled(x)` (0 for a side the row does not have), acts on `.committed(.leading | .trailing)`
(leading is a drag to the right: Done; trailing is a drag to the left: Not now, Remove, To get) and brings the row home
on `.cancelled`. `.ended` is always last: that is where a row clears "a swipe just happened, so the tap that follows
is not a tap". If the system takes the touch (a scroll wins), the row still gets `.cancelled` and `.ended`.

## Action words, wheels and stacks (the patterns that are coming)

- **Action word** (the bottom-left word): lean at 18 pt (`actionLean`), roll to the option's name past 56 pt
  (`actionRoll`), commit on release with the option's haptic and Undo for 10 s; `GestureTuning.actionWordStage` and
  `actionWordCommits` are the whole rule. The word starts at least 28 pt from the leading edge. A tap on it is still a
  tap (10 pt slop).
- **Depth wheel:** drag starts at 10 pt, one detent per row pitch, a snap on release with no momentum, a tick per
  detent. A tap on a row is ignored for 120 ms after the wheel lets go (`TapHoldoff`). Only the focused row's button
  is live under a drag.
- **Home mosaic widget stack:** commits a swipe past 35% of its width (`stackCommitFraction`,
  `GestureTuning.stackCommits`), or on a flick; the same axis lock and edge rules.
- **Reduce Motion and Low Power** change the motion, never the thresholds.

## What was wrong (the audit, Oct 24)

About 100 files carry gestures. The same intent was written down 17 ways:

- **Drag start distance** ran from 0 to 30 pt: 0 (13 sites), 4 (the Places dial), 6 to 8 (five), 10 to 16 (ten), 20 (six),
  24 (fifteen), 28 and 30 (five). A 4 pt dial and an 8 pt unit swipe start from a jitter; a 30 pt drag feels stuck.
- **Axis** was checked in `onChanged` on every sample, each time on a different rule (`|dx| > |dy|`, `>= 1.5x`, `2x`,
  `<60 pt`), and a swipe that had started could be cancelled, or a scroll converted, by later drift. Several rows
  committed on release from a pull the last sample had stopped drawing.
- **Commit** distances: 48, 60, 72, 80, 90, 96, 120, 40% of a column. Flick rules: none, or 12 pt at 300 pt/s (a
  twitch turned a Home page). A swipe taken back at lift still committed.
- **Unit swipe** (`MeasureView`): any 22 pt vertical drag that began on a number changed its unit (8 pt start plus a
  14 pt step), so a scroll over a number changed units.
- **Small pulls** opened things from a 20 to 30 pt drag: the running order (30 pt), the effort word (20 pt), the swap
  panel (44 pt).
- **Pickers by drag with no lead**: the Routines kind word (16 pt, `|dx| > |dy|`), Training log months (30 pt), Settings
  tabs (24 pt) flipped on a diagonal.
- **Back edge:** right swipes (Done on Tasks, the Shop run strike, Routines start, Lists "check off next", the page
  turns) started anywhere, including the 24 pt the back gesture owns, so a back swipe could finish a task.
- **Holds** came in eight lengths (0.18, 0.2, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7 s) with five slops (6, 8, 10, 20, 30 pt).
- **Haptics on arm** were played by some rows and not others; two used a private flag.
- **highPriorityGesture** (3): the apps list pick-up (dead taps in release 77, Builder B is fixing it), Home's one
  thing (intended: sheet 220), the exercise sheet's long press. **simultaneousGesture** (37) is the right tool for
  row swipes; the mistakes were in what they did inside.
- **Taps after a wheel** (the Places dial, the apps list) were read as picks.
- **No test** held the line that a plain tap reaches its button in a list with a drag on it.

## What this change moved onto the policy

`rowSwipe`: Tasks (both ways), Pantry and the Places rail (To get), Lists (Remove, check off next, Skip a pick),
Routines (Drop, Start), Recipes (the column to Groceries), the Shop run strike, the proposed-capture drop, the cook
timer stop. `pageSwipe`: Settings and Review tabs, Training log months, the Routines kind word, the You door's edge,
and the mosaic pager's rule. `pullGesture`: the workout's running order, the effort word, the swap panel and its
close, the finished screen, Tess's room and conversation, the capture wash, the exercise strip, the cook step
(through `GestureTuning.pull`). Also: Home's one thing (distance, flick and a diagonal ignored), the Places dial (10
pt start, 120 ms tap hold-off) and the unit swipe (2x vertical lead, 28 pt per step).

## Every distance that changed (for Design to tune)

Consistency was the ask, so these stand as the policy sets them. Tune a number in `GestureTuning`, not at a site.

| Where | Before | Now |
| --- | --- | --- |
| Tasks Done / Not now | arm 120, 20 pt start | arm 96, 10 pt start, flick |
| Pantry and Places rail, To get | arm 72, 24 pt start | arm 96, 10 pt start |
| Lists Remove / Forget | arm 96, 24 pt start | arm 96, 10 pt start, flick |
| Routines Drop, Start | arm 90, 20 pt start | arm 96, 10 pt start |
| Recipes column to Groceries | 40% of the column, 24 pt start | 40%, 10 pt start, flick |
| Shop run strike | arm 72 | arm 96 |
| Proposed capture drop | 80 pt, 12 pt start | arm 96, no flick (no Undo) |
| Lists check off, Skip a pick, cook timer stop | 72, 60, 80 | 72 (undrawn), flick |
| Pull down: running order (EffortHandle and StepStack included) | 30 pt | 56 pt or flick |
| Pull: effort word, swap panel | 20 pt, 44 pt | 56 pt or flick |
| Close: finished screen, Tess room, conversation, capture wash, exercise strip | 80, 60, 70, 40, 40 | 56 pt or flick |
| Cook step (up / down) | 60 pt, 24 pt start | 56 pt or flick, 10 pt start |
| Page turn: Settings and Review tabs | 40 pt at 2:1 | 60 pt or flick, 2:1 |
| Page turn: Training log, Routines kind word, You door edge | 30 pt, 16 pt, 48 pt | 60 pt or flick, 2:1 |
| Unit swipe on a number | 8 pt start, 14 pt a step, any vertical lead | 10 pt start, 28 pt a step, 2:1 lead |
| Places dial | 4 pt start | 10 pt start, 120 ms tap hold-off |
| Home one thing | 12 pt start, flick = 12 pt at 300 pt/s, any diagonal | 10 pt start, flick = 28 pt at 500 pt/s, diagonal ignored |

Rows with an Undo (Tasks, Lists, Forget, Routines drop, cook timer stop, Recipes send) accept a flick; the capture drop
has none, so only a full pull commits it.

## What it left alone, on purpose

- The Home apps list (`AppsListSlot`): Builder B is hot-fixing the dead taps; its wheel should move onto
  `dragStart`, `TapHoldoff` and `GestureTuning.commits` once that lands.
- The Today queue and plan rows (`RowPressDrag`, a UIKit hold-then-drag that never claims a moving finger): it is
  already the model (hold 0.2 s, 8 pt slop, commit by distance); its numbers should come from here next.
- The instruments (weight reel, reps, effort stops, rest horizon) and the soul's and mosaic's touch feeds: they use
  `minimumDistance: 0` deliberately and lock the axis at 10 pt (`ScrubFeel`).
- The doorway back swipes (`startLocation.x < 28`, 80 pt): they stand in for the system back edge on covers.

## Review checklist (QA)

- G13: every list with a hold or drag gesture lets a plain tap reach its button, verified by a UI test
  (`GestureStabilityUITests` runs it in every room that opens by `-openLink`).
- New code: no `DragGesture(minimumDistance: 0)` outside an instrument; no hand-written axis check; no private
  distances; every swipe or pull through `rowSwipe`, `pageSwipe` or `pullGesture`; a rightward action starts 28 pt in;
  every commit through `ActionEngine` so Undo is there; a VoiceOver action for each swipe.
- A new threshold goes into `GestureTuning` with a test first.
