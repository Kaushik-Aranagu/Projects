# Training build, sheet 230 (Design handoff, Oct 23-24, all numbers verbatim from relays)

Board https://claude.ai/artifact/PFDdUde9cLtYA1G6Qjrpwp (v14). The sheet wins on any doubt. Files: Features/Landing (Training landing), Features/Workout/WorkoutModeView.swift, LiveSetMarks.swift, SetMarkModel.swift, ExerciseStrip.swift (becomes the wheel), ExerciseSwaps.swift, Features/Exercise.

## Decisions that still stand
- Colors: cobalt 0B62AE light / 8FB0EE dark (Soft palette, already in AppAccents on branch ios/a-night-1). Go green 2FA24D light / 8FC9A0 dark: large text and strokes only (warm-up, rest countdown, set strike, "Done warming up", "Skip rest"), never the only signal. Soft dark tokens on pure black 000000: accent 8FB0EE, go green 8FC9A0, ink E8E3DA, ink2 A49E96, ink3 78736D, edge 2A2A2E. Light unchanged.
- NO decorative underlines anywhere (today's week letter is cobalt 600, no underline, VoiceOver says "today"; minutes value and unit word have none).
- NO pixel art outside the Home mosaic (no lintel, no ridge). NO glass lens (cancelled). No ring, circle or ink D set mark remains.
- Big fixed titles shrink 1 pt at a time until they fit, down to 0.75x. Exercise name 34 down to 26 at most; only past that floor Tess's short form (over 22 chars) takes over.
- Wheel blur follows ONE shared setting Settings > Look > "Depth on wheels" (default On; B builds the same switch for Notes: check master for it, else add `ReviewChoices`/settings key "depthOnWheels" and tell the coordinator). Blur rendered once per row statically, never a live per-frame blur. No blur under Reduce Motion, Reduce Transparency or Low Power.
- One shared wheel component for plan, session, week, library and the number dial, built on GestureTuning/GestureKit (on master).

## Part 1: rep marks and set animation
- Set mark = STACK AND HANDLE, one cable-machine stack per set. 11x24 pt (12x28 at the focus), 5 pt between sets. Parts top to bottom: 5.2 pt grip at 1.6 on a short cable; the cap (top plate) at 1.4; a rod at 0.8 in ink3; a floor at 1 in ink3. One plate line per rep. Logged reps lift up under the cap in go green at 1.15. Reps still to do rest on the floor in ink3 at 0.8. The rod stops at the first plate it carries; when nothing is lifted it stops at the top resting plate; it never runs to the floor. Cable and cap turn green once a set has any reps; the grip turns green when the set is full.
- Sets log WHOLE at the planned reps. No per-rep logging. The foot "Log set N" logs the full target. After logging the brief confirms for 5 s: "Set 1, 5 reps." where the 5 SCROLLS (no - + steppers anywhere); any change restarts the 5 s; when it runs out it reads "Set 1, 5 reps, kept." (a confirmation, not a question). Count range 0 to planned+10. Undo 10 s: "Set 2, logged. Undo".
- Animation order on a logged set: stack plates lift in order, 70 ms apart, green strike draws across, they gather to the rep number (Outfit 200, 44 pt), the number arcs to the trail on the right and shrinks to 26 pt (Outfit 300 26 pt, like "5 5") with a small bounce, "Set 1 of 3" rolls its digit, then rest starts. Reduce Motion: the trail number simply appears.
- Tapping a lift's stack in either wheel opens its lift page.

## Scrolling numbers (no steppers anywhere)
- Drag up/down on a number: 16 pt per step; weight steps by 5 (lb) or 2.5 (kg). Mouse/trackpad wheel also scrolls. Light selection tick per value. While touching, next and previous values show faintly above and below. Tap opens a dial: a full-room depth wheel of values with the same row/blur rules as the plan; Done at bottom-left; Esc or Done keeps the focused value. Each number is an accessibility adjustable element (increment/decrement).
- Applies to the 5 s set confirmation and to Sets, Reps, Weight on the lift page.

## Part 2: top band and rest (workout mode)
- Top band is only the current lift. Name 34 pt 600 cobalt in a one-line slot (shrink rule above). Tap opens the exercise page. "Set 1 of 3" at a fixed width. A "was 185" span in ink2 appears 2 s after a weight change. Weight Outfit 200 84 pt with the unit word (22 pt 600 cobalt, NO underline) beside it. Units shown "185 lb" with the unit at half size on wheel rows: Figtree 400, 7.5 pt, ink2, 2 pt after the numeral.
- Weight changes: drag the number in 5 lb / 2.5 kg steps; tap or swipe the unit to switch (digits roll; receipt "185 lb loads as 85 kg."). Number, bar and wheel have separate hit areas. No vertical page turn in workout mode.
- Rest: "Rest 1:30" in green then "Set 2 is next", foot "Skip rest". When rest ends: Live Activity update + local notification, sound follows the silent switch.
- Warm-up: "Rower, easy" in green with 8:00 countdown and "Keep it easy, about 22 strokes a minute. Bench is next." Foot "Done warming up".
- Done: "Push A", the time, "warm-up and 4 lifts", "Bench held at 185. Next Push A is Friday." Foot "Finish" with Undo 10 s.
- The foot always holds ONE word: Start Push A, Done warming up, Log set N, Skip rest, Finish, Pick up Push A / Resume Push A.

## Session wheel (inside the workout) and plan wheel
- Bottom, above the 36 pt foot band. Five 34 pt rows, focus on the middle. Current lift 22 pt 600 cobalt with "now"; neighbours 16 pt ink2 then 14 pt ink2 weight 400, every row ink2 or darker. Browsed row ink 600 no "now". Info: done lifts show their trail ("5 5 5"), next "3 by 6", part-done "5, 2 left". (Set values on rows become the stack mark.)
- Landing plan wheel (v7): focused row top slot 30 pt Semibold cobalt (warm-up uses go green); others 20, 17, 15 pt ink2; spacing 38 pt plus extra 8 pt next to the focused row. Blur 0.65 pt one away, 1.0 pt two or more, never more than 1 pt. Header row (row -1, scrolls away) small caps 13 pt weight 600 ink2, no line: "The plan" + "4 lifts after the warm-up" (landing); "This session" + "N lifts" (workout). Long focused names shrink to fit beside their set marks. Finished lifts get a 1 pt hairline strike in go green through the name (readable); skipped lifts strike in ink3 (new skipped state kept in L.skip, reversible by the 10 s Undo).
- Gestures (220): hold-scroll tracks the finger 1:1, a flick moves exactly one row with spring snap and one light haptic per row; tapping another row centres it; tapping the focused row opens its exercise page. Springs back to now 6 s after the last touch (each touch resets; pauses while a finger is down, VoiceOver focus on the wheel, or an exercise page is open). AX3+: 3 rows. VoiceOver reads it as a list. CAUTION: a plain tap must still reach the row buttons (see the apps-list tap bug: a highPriorityGesture LongPress+Drag(0) swallowed taps). Add UI tap tests.
- Exercise page: slides in from the right, left edge back. "Works chest, shoulders and triceps", name, Gym visual GIF loop (still frame under Reduce Motion or Low Power), best, last 3 sessions (history trimmed to 2 lines), next time. For a lift not done: foot "Do it now" swaps it in; receipt "Dips now, bench press after. Undo" (10 s). Lift page also gets a sets block above Best ("Today's sets" in a workout, "Usual sets" otherwise; rows Sets, Reps, Weight as scrolling numbers; Sets can't go below the sets already done; in a workout the line "Also the usual: No/Yes"; wheels update live). Sets/reps/weight rows use scrolling numbers.

## Left-edge back, action word
- Left-edge back from the workout keeps it running. The landing reads "Today, going"; brief "Bench press, set 2 of 3. 4 lifts to go."; action word "Resume Push A".
- Action word gestures (rule for every room): tap does the action; dragging left or right offers other options, no menus. Word leans 18 pt max; past 56 pt it rolls over to the option's name with a selection tick; releasing past that commits (.medium impact, 10 s Undo), else springs back. Left = stop/remove, right = again. Each option also an accessibilityAction (and reachable by plain taps for Switch Control). Touch start must be at least 28 pt from the left edge; roll only takes priority when the touch starts on the word itself so the back swipe still works. Training not started: left "Skip today". Started: left "End workout", right "Start over". Inside the workout: left "Skip warm up" / "Skip this lift" ("End workout" on the last lift), right after a set "Same again". Every committing option needs a TRUE Undo ("End workout" restores the running workout, "Skip" restores the lift); size the option slot for the longest word, exactly one visible.

## Part 4 landing v2 (build now; LIBRARY HELD, design is making it a matrix)
- New landing with fixed slots: "Training" title 40 pt. Three stat lines: "3 of 6 this week", "5 weeks in a row on plan", "+10 lb on bench since Oct 1" (gain in go green); tapping a stat line opens that lift. A 118 pt today slot. Two doors "This week" and "Library", each a 26 pt word with one ink2 line under it.
- Today slot states: Not started "Push A" in the accent (tap opens today's workout). In progress: green "In progress" header, the clock, a "Pause" word, and where he is ("Bench press, set 2 of 3."). Paused: clock stops, turns ink3, word reads "Go on". Done and Skipped: Push A with a 1 px strike.
- Action word: "Start Push A" or "Resume Push A". Drag left Skip today / End workout; drag right Pause / Go on.
- Today's workout = the old landing moved one level in, without the weekday strip; pages slide in from the right, left edge or Esc goes back. Resume: leaving mid-workout keeps the workout; foot reads "Pick up Push A"; Home and the Live Activity open the same place. A private workout shows "Private workout" in the name slot, the Live Activity and the Lock Screen. The landing never jumps into the workout and shows no weight or bar.
- This week: the shared depth wheel centred on today. Today keeps the accent even off focus. Done names get a 1 px green strike and green minutes; rest rows ink3. The brief follows the row in front; tapping today opens it. Action word "Move Push A": a tap slides it a day later with Undo; drag left to skip.
- Week-letter bar moves off the top. Stats come from Core's TrainingLedger API (Core is adding it); use sample values behind a clear seam until it lands.

## Accept checks
1 No ring, circle or ink D remains. 2 Reduce Motion: nothing moves. 3 A flick moves one row; a drag on the number never moves the wheel. 4 Spring-back waits 6 s and pauses while an exercise page is open. 5 Green only on large text and strokes, always with words. 6 The foot shows one word and Tess never covers the trail. 7 Undo restores a set, a swap and Done within 10 s. 8 At AX3 the wheel shows 3 rows and a name over its floor uses the short form. One PR per part to QA, all On.

## Core API for the landing (landed)
All on `TrainingStats(store:classes:)` (engine only, pure reads), words in `TrainingSentences`:
- `weekCount(plan:now:calendar:) -> WeekCount(done, planned)` and `TrainingSentences.weekCount(_:)` -> "3 of 6 this week".
- `weeksOnPlan(plan:now:calendar:) -> Int` and `TrainingSentences.weeksInARow(_:)` -> "5 weeks in a row on plan" (nil under 2).
  A week counts when its sessions reached what the plan asks for (`sessionsPerWeek(of:now:)`: the plan's number, else one per
  routine up to six; on fixed days, the days that have a routine). This week counts once met and never breaks the run
  while it is being done.
- `headlineGain(since:now:unit:calendar:) -> LiftGain?` and `TrainingSentences.gain(_:calendar:)` -> "+10 lb on bench press
  since Oct 1" (nil unless a lift is up; `LiftGain.exerciseID` is what the tap opens). Shorten the exercise name in the
  app (Tess's short form), not here.
- Warm-up: three fields on the routine; `warmUp(of:) -> WarmUp?` (name, minutes, note; nil for none or a private
  routine), `WarmUp.phrase` -> "8 min on the rower", `TrainingSentences.warmUpLine(_:)` -> "Starts with 8 min on the
  rower", and the action `SetWarmUp(routine:name:minutes:note:)` (an empty name takes it off; refuses minutes outside 1
  to 30 and a name or note that holds a secret word, a body fact or a key).
- Skip today / Move Push A: a rest day (`TakeRest(scope: .training)`, from the element actions). A rolling plan never
  carries debt, so the next routine simply comes next; the receipt and the 10 s Undo are the action's own.
- Held with the Library: goals, programs and the matrix data (design is still making it).
