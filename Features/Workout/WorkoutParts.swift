import SwiftUI
import TesseraCore

private typealias Paint = WorkoutPaint

// MARK: - The exercise lane (x)

/// One name per page; nothing peeks (the strip says what's ahead). Tap the name to zoom into the exercise. Pull down on
/// it for the swap panel:
/// the name follows the finger (rubber-banded to about 10 pt). The pager keeps horizontal.
struct ExerciseLane: View {
    let session: WorkoutSession
    @Binding var laneIndex: Int?
    let isSwapOpen: Bool
    var setSwap: (Bool) -> Void

    @State private var namePull: CGFloat = 0
    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.zoomNamespace) private var zoomNamespace

    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        ScrollView(.horizontal) {
            LazyHStack(spacing: 0) {
                ForEach(Array(session.entries.enumerated()), id: \.element.id) { index, entry in
                    let exercise = session.exerciseClass(of: entry)
                    let isCurrent = index == (laneIndex ?? session.currentIndex)
                    let name = exercise?.name ?? "Exercise"
                    let muscles = MuscleReading.muscles(of: exercise, store: model.store)
                    Button {
                        if let exercise {
                            model.open(.classDef(exercise.id), .deeper)
                        }
                    } label: {
                        // Title 44 on one line, else 34 on one line, else two lines of 34. Never cut, and never drawn
                        // past its own page.
                        ViewThatFits(in: .horizontal) {
                            Text(name).textRole(.title).lineLimit(1).fixedSize()
                            Text(name).font(Self.title34).lineLimit(1).fixedSize()
                            Text(name).font(Self.title34).lineLimit(nil)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .foregroundStyle(Paint.cream)
                        .opacity(isCurrent ? 1 : 0)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id(name)
                        .transition(reduceMotion ? .opacity
                            : .asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity),
                                          removal: .opacity))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .modifier(ZoomSource(id: exercise.map { Route.classDef($0.id) }, namespace: zoomNamespace))
                    .containerRelativeFrame(.horizontal)
                    // Nothing draws past the page's left or right edge; the height is left alone (large text wraps).
                    .mask { Rectangle().padding(.vertical, -2000) }
                    .id(index)
                    .offset(y: isCurrent ? Self.rubberBand(namePull) : 0)
                    .contextMenu {
                        if isCurrent {
                            Button("Swap within the movement", systemImage: "arrow.left.arrow.right") {
                                setSwap(true)
                            }
                        }
                    }
                    .accessibilityLabel([
                        name,
                        session.shownTitle != session.workout.title ? "" : MuscleReading.spoken(muscles),
                    ]
                    .filter { !$0.isEmpty }
                    .joined(separator: ", "))
                    .accessibilityHint(isCurrent ? "Opens the exercise page" : "")
                    .accessibilityAction(named: "Swap within the movement") { setSwap(true) }
                    .tesseraAnimation(.easeOut(duration: 0.24), value: isCurrent)
                }
            }
            .scrollTargetLayout()
        }
        .contentMargins(.leading, Theme.Spacing.gutter, for: .scrollContent)
        .contentMargins(.trailing, Theme.Spacing.gutter, for: .scrollContent)
        .scrollTargetBehavior(.viewAligned(limitBehavior: .alwaysByOne))
        .scrollPosition(id: $laneIndex, anchor: .leading)
        .scrollIndicators(.hidden)
        // Large text wraps the name to three lines: the pager keeps room for them (it measures one page at a time).
        .frame(minHeight: typeSize.isAccessibilitySize ? 210 : 64)
        .fixedSize(horizontal: false, vertical: true)
        // Pull down on the name for alternatives; the pager keeps horizontal.
        .simultaneousGesture(
            DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart))
                .onChanged { drag in
                    let lead = CGFloat(GestureTuning.pagerBias)
                    guard session.finishedAt == nil, drag.translation.height > 0,
                          abs(drag.translation.height) >= abs(drag.translation.width) * lead
                    else { return }
                    namePull = drag.translation.height
                }
                .onEnded { drag in
                    let pull = GestureTuning.pull(
                        dx: Double(drag.translation.width), dy: Double(drag.translation.height),
                        velocityY: Double(drag.velocity.height)
                    )
                    withTesseraAnimation(Theme.Motion.snappy, reduceMotion: reduceMotion) { namePull = 0 }
                    if pull > 0 {
                        setSwap(!isSwapOpen)
                    } else if pull < 0, isSwapOpen {
                        setSwap(false)
                    }
                }
        )
    }

    /// Title 34: the title face one step down, for names past about 16 characters.
    private static var title34: Font {
        var spec = Theme.concept.scale.title
        spec.size = 34
        return Theme.Typography.font(spec)
    }

    /// The name follows the finger up to about 10 pt, rubber-banded.
    private static func rubberBand(_ pull: CGFloat) -> CGFloat {
        guard pull > 0 else { return 0 }
        return 10 * (1 - exp(-pull / 40))
    }
}

// MARK: - Lines

/// The subline and Tess's line, in words, from the session's state.
@MainActor
enum WorkoutLines {
    static func subline(_ session: WorkoutSession, phase: WorkoutSession.Phase) -> (String, Color, wet: Bool) {
        // `wet`: the line arrives in cadmium and dries to the context colour (sweep-19).
        guard let entry = session.currentEntry else { return ("", Paint.secondary, false) }
        if let row = session.editingRow {
            return (
                "set \(row.number), as logged: \(Format.kilograms(row.weight)) × \(Int(row.reps))",
                Paint.tess,
                false
            )
        }
        let target = session.targetSets(of: entry)
        let rows = session.setRows(of: entry)
        let tenth = { (value: Double) in Format.kilograms((value * 10).rounded() / 10) }
        switch phase {
        case .rest:
            if let seated = session.lastSeated, !seated.records.isEmpty, let e1RM = seated.e1RM {
                return ("a new best: e1RM \(tenth(e1RM))", Paint.secondary, false)
            }
            // The exercise's last rest sums it up: "4 sets, 82.5 × 8 best".
            if rows.count >= target, let best = bestRow(rows) {
                return ("\(rows.count) sets, \(best) best", Paint.secondary, false)
            }
            // After a set: "set 1, 82.5 × 8".
            if let last = rows.last {
                return ("set \(last.number), \(Format.kilograms(last.weight)) × \(Int(last.reps))", Paint.secondary,
                        false)
            }
            return ("", Paint.secondary, false)
        case .exerciseDone, .workoutDone:
            if let best = bestRow(rows) {
                return ("\(rows.count) sets, \(best) best", Paint.secondary, false)
            }
            return ("", Paint.secondary, false)
        case .set:
            if let rpe = session.rpe {
                return ("RPE \(RPEStops.formatted(rpe)), \(RPEStops.words(for: rpe))", Paint.secondary, false)
            }
            if let hint = session.recordHint {
                let previous = hint.previous.map { ", \($0)" } ?? ""
                return ("set \(session.currentSetNumber): \(hint.text)\(previous)", Paint.secondary, false)
            }
            if let hint = session.swapHint, rows.isEmpty {
                return (hint, Paint.secondary, false)
            }
            // "last time 82.5 × 8": this set's place last session.
            if let last = session.lastTime(setNumber: session.currentSetNumber) {
                return (
                    "last time \(Format.kilograms(last.weight))\u{00A0}×\u{00A0}\(Int(last.reps.rounded()))",
                    Paint.secondary,
                    false
                )
            }
            return ("first time on this one", Paint.secondary, false)
        }
    }

    /// The heaviest set, "82.5 × 8".
    private static func bestRow(_ rows: [WorkoutSession.SetRow]) -> String? {
        rows.max { ($0.weight, $0.reps) < ($1.weight, $1.reps) }
            .map { "\(Format.kilograms($0.weight)) × \(Int($0.reps))" }
    }

    /// The context line above the name (live-workout.md): "Push A, set 1 of 4"; resting "Rest, then set 2" or, on an
    /// exercise's last rest, "Rest, then Overhead press"; the workout's last set "Push A, last set".
    static func context(_ session: WorkoutSession, phase: WorkoutSession.Phase) -> String {
        let title = session.shownTitle
        guard let entry = session.currentEntry else { return title }
        let target = session.targetSets(of: entry)
        let done = session.setRows(of: entry).count
        switch phase {
        case .rest:
            if done >= target {
                if let next = session.nextUnfinishedIndex, session.entries.indices.contains(next),
                   let name = session.exerciseClass(of: session.entries[next])?.name
                {
                    return "Rest, then \(sentenceCase(name))"
                }
                return "Rest, then finish"
            }
            return "Rest, then set \(done + 1)"
        case .exerciseDone:
            return "\(title), \(sentenceCase(session.currentExercise?.name ?? "this one")) done"
        case .workoutDone:
            return "\(title), every set in"
        case .set:
            if session.isLastSetOfWorkout {
                return "\(title), last set"
            }
            return "\(title), set \(session.currentSetNumber) of \(max(target, 1))"
        }
    }

    /// "Overhead Press" reads "Overhead press" inside a line; all-capital words ("DB", "EZ") stay as written.
    static func sentenceCase(_ name: String) -> String {
        name.split(separator: " ", omittingEmptySubsequences: false).enumerated().map { index, word in
            let text = String(word)
            if index == 0 || (text.count > 1 && text == text.uppercased()) {
                return text
            }
            return text.lowercased()
        }.joined(separator: " ")
    }

    static func tess(_ session: WorkoutSession, phase: WorkoutSession.Phase, summary: String?) -> String {
        if session.finishedAt != nil {
            return summary ?? ""
        }
        if session.isEditing {
            return "Dial it in, then save."
        }
        if let note = session.editNote {
            return note
        }
        if case .rest = phase, let seated = session.lastSeated {
            // The receipt (live-workout.md §6): "Set 3 logged. 80 × 8." and "New best: 85 × 5." (no e1RM in a
            // receipt; the estimate lives on the exercise page, Product 2:29 PM PT Oct 16).
            let set = "\(Format.kilograms(seated.weight)) × \(Int(seated.reps))"
            if seated.records.isEmpty {
                return "Set \(seated.number) logged. \(set)."
            }
            return "New best: \(set)."
        }
        switch phase {
        case .exerciseDone: return "\(session.currentExercise?.name ?? "That") is done."
        case .workoutDone: return "Last one. Hold to finish."
        default:
            return setLine(session)
        }
    }

    /// The receipt after a set lands, thing first (receipts-thing-first.md RC10): "Set 3 logged.", else "Set, logged.".
    static func loggedLine(_ number: Int?) -> String {
        number.map { "Set \($0) logged." } ?? "Set, logged."
    }

    /// While dialling a set.
    private static func setLine(_ session: WorkoutSession) -> String {
        if let from = session.swappedFrom, let entry = session.currentEntry, session.setRows(of: entry).isEmpty {
            return "\(SwapPanel.short(from)) stays in the plan."
        }
        // Nothing is said about a best before the set; it's said once logged ("New best: …", 56 addendum).
        if let note = session.restNote, note.hasPrefix("Rest skipped") {
            return note
        }
        if session.canUndo {
            return loggedLine(session.lastSeated?.number)
        }
        if let reason = session.targetReason, session.currentSetNumber == 1 {
            return reason + "."
        }
        // Before a set the page already says it all (the key, the detail line): Tess stays quiet (56 addendum).
        return ""
    }
}

// MARK: - Effort handle

/// Pull up on the reps (or tap "reps") to slide the effort stop-slider out; pull down to put it away.
struct EffortHandle: View {
    let isShown: Bool
    let enabled: Bool
    var setShown: (Bool) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text("reps").font(Theme.voice(size: 16)).foregroundStyle(Paint.cream.opacity(0.55))
            Text("effort")
                .textRole(.label)
                .foregroundStyle((isShown ? Paint.now : Paint.secondary).opacity(0.7))
        }
        .frame(minWidth: 92, minHeight: 44)
        .contentShape(Rectangle())
        .onTapGesture {
            if enabled {
                setShown(!isShown)
            }
        }
        .pullGesture { down in
            guard enabled else { return }
            setShown(!down)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(isShown ? "Hide effort" : "Add effort")
        .accessibilityAddTraits(.isButton)
        .accessibilityAction {
            if enabled {
                setShown(!isShown)
            }
        }
    }
}

// MARK: - Edit keys

/// While a logged set is on the dial: Save set N (Core's EditSet) and a plain Cancel.
struct EditSetKeys: View {
    let session: WorkoutSession
    var onSave: () -> Void
    var onRemove: () -> Void = {}
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if let row = session.editingRow {
            // A tap in that set's own paint; the meta follows the dial and drum. Silent: onSave plays the set's tap.
            PaintKey(paint: Paint.set(row.number), isTap: true, height: 64, isSilent: true) {
                withTesseraAnimation(Theme.Motion.snappy, reduceMotion: reduceMotion) { session.saveEdit() }
                onSave()
            } label: { _ in
                Text("Save set \(row.number)")
            } meta: {
                Text("\(Format.kilograms(session.weight)) × \(Int(session.reps))")
            }
            .accessibilityLabel("Save set \(row.number) as \(Format.kilograms(session.weight)) by \(Int(session.reps))")
            Spacer(minLength: 0)
            HStack(spacing: 30) {
                Button("Remove") {
                    withTesseraAnimation(Theme.Motion.snappy, reduceMotion: reduceMotion) { session.removeEditing() }
                    onRemove()
                }
                .buttonStyle(.labelWord(Theme.Colors.liveInk))
                .accessibilityLabel("Remove set \(row.number)")
                Button("Cancel") {
                    withTesseraAnimation(Theme.Motion.snappy, reduceMotion: reduceMotion) { session.endEditing() }
                }
                .buttonStyle(.labelWord(Paint.secondary))
            }
        }
    }
}

// MARK: - Typing to Tess in live mode

/// Typed capture during a workout ("log bench 80x8", "8", "same"): a field that sits on the keyboard, sends the
/// phrase through Core's capture engine (the set shows as a ghost stroke, then seats), and closes.
struct LiveAskField: View {
    @Binding var isShown: Bool
    var onAsk: (String) -> Void
    /// A key was cleared: say where it goes, as Ask does.
    var onRefuse: () -> Void = {}
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Spacing.s) {
            TextField("log bench 80x8, or ask", text: $draft)
                .textRole(.field)
                .foregroundStyle(Paint.cream)
                .focused($focused)
                .submitLabel(.go)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .onSubmit(submit)
                // A Claude key pasted here is cleared the moment it lands: no ghost, no set, no miss log.
                .onChange(of: draft) { _, text in
                    if LiveAskField.refuses(text) {
                        draft = ""
                        onRefuse()
                    }
                }
            Button("Close") { close() }
                .buttonStyle(.labelWord(Paint.secondary))
        }
        // A writing line, not a field: 0.5 pt under the words (no-buttons.md, "Write").
        .padding(.bottom, 6)
        .overlay(alignment: .bottom) { Rectangle().fill(Paint.tess.opacity(0.6)).frame(height: 0.5) }
        .padding(.horizontal, Theme.Spacing.gutter)
        .padding(.top, Theme.Spacing.s)
        .frame(maxHeight: .infinity, alignment: .top)
        .preferredColorScheme(ThemeStore.shared.workoutLook.isDark ? .dark : .light)
        .onAppear { focused = true }
    }

    /// Anything that looks like a Claude API key, in any case (the guard Ask uses, BottomChrome.looksLikeClaudeKey).
    static func refuses(_ text: String) -> Bool {
        looksLikeClaudeKey(text)
    }

    private func submit() {
        let phrase = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        draft = ""
        if !phrase.isEmpty, !Self.refuses(phrase) {
            onAsk(phrase)
        }
        close()
    }

    private func close() {
        focused = false
        isShown = false
    }
}
