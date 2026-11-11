import CryptoKit
import Foundation

/// Built-in classes every install starts with. Seeding is idempotent, and system classes and fields
/// get deterministic ids derived from their keys, so separate devices seed identical `id`s.
/// CloudKit does not merge records by our `id`, though: two devices that both seed will sync two copies.
/// A dedupe pass by `systemKey` / field id must run before sync is switched on (see docs/engine/research-01.md).
@MainActor
// The catalog is mostly declarative class specs; their length is the point.
// swiftlint:disable:next type_body_length
public enum SystemCatalog {
    public enum Key: String, CaseIterable, Sendable {
        case thing, activity, exercise, strengthExercise, barbellLift, benchPress
        case muscle, equipment
        case workout, exerciseEntry, set
        case routine, task, note
        case list, listItem
        case ingredient, recipe, recipeLine, recipeStep, cook, kitchenTimer
        case cookSession, cookDish
        case process
        case slot, plan
        case tessCall
        case event, sleepNight, stepsDay, restingHeartRate, closedDay, heldNotice, spend, book
        case pool, inflow, poolMove, rateNote
        case tessSent
        case tessSentExact
        case settings
        case meProfile
        case place
        case tessWaiting
        case tessMemory
        case occasion
        case tessTurn
        case belief
        case routineRun, rest
        case tessAgent, notebookPage, agentEpisode
        case room, roomPart

        public var systemKey: String {
            "system.\(rawValue)"
        }
    }

    private struct FieldSpec {
        var key: String
        var name: String
        var kind: FieldKind
        var unit: String?
        var defaultValue: FieldValue?
        var target: Key?
        var options: [String] = []
        var expression: String?
        /// Overrides the default of an inherited field with this key instead of declaring a new one.
        var overrides: String?
    }

    private struct ClassSpec {
        var key: Key
        var name: String
        var parent: Key?
        var symbol: String
        var colorRole: String
        var traits: [ClassTrait] = []
        var fields: [FieldSpec] = []
    }

    /// Order matters: parents before children, link targets before fields that use them.
    private static let specs: [ClassSpec] = [
        ClassSpec(key: .thing, name: "Thing", symbol: "circle", colorRole: "neutral", fields: [
            FieldSpec(key: "notes", name: "Notes", kind: .richText),
            FieldSpec(key: "notTwins", name: "Not the same as", kind: .refList, target: .thing),
        ]),
        ClassSpec(key: .muscle, name: "Muscle", parent: .thing, symbol: "figure.arms.open", colorRole: "training",
                  traits: [.measurable], fields: [
                      FieldSpec(key: "group", name: "Group", kind: .text),
                      FieldSpec(
                          key: "weeklySetsMin",
                          name: "Weekly sets from",
                          kind: .number,
                          defaultValue: .number(10)
                      ),
                      FieldSpec(key: "weeklySetsMax", name: "Weekly sets to", kind: .number, defaultValue: .number(20)),
                  ]),
        ClassSpec(key: .equipment, name: "Equipment", parent: .thing, symbol: "dumbbell", colorRole: "training",
                  fields: [
                      FieldSpec(key: "kind", name: "Kind", kind: .choice,
                                options: EquipmentKind.allCases.map(\.rawValue)),
                      FieldSpec(key: "barWeight", name: "Bar", kind: .mass, unit: "kg"),
                      // Plate sizes available in pairs, e.g. "25, 20, 15, 10, 5, 2.5, 1.25".
                      FieldSpec(key: "plates", name: "Plates", kind: .text),
                      // Fixed loads for dumbbells or a machine stack, e.g. "2, 4, 6, 8, 10, 12.5".
                      FieldSpec(key: "loads", name: "Loads", kind: .text),
                      FieldSpec(key: "increment", name: "Smallest step", kind: .mass, unit: "kg"),
                  ]),
        ClassSpec(key: .activity, name: "Activity", parent: .thing, symbol: "figure.walk", colorRole: "training"),
        ClassSpec(key: .exercise, name: "Exercise", parent: .activity, symbol: "figure.strengthtraining.traditional",
                  colorRole: "training", fields: [
                      FieldSpec(key: "primaryMuscles", name: "Primary muscles", kind: .refList, target: .muscle),
                      FieldSpec(key: "secondaryMuscles", name: "Secondary muscles", kind: .refList, target: .muscle),
                      FieldSpec(key: "equipment", name: "Equipment", kind: .refList, target: .equipment),
                      FieldSpec(
                          key: "restSeconds",
                          name: "Rest",
                          kind: .duration,
                          unit: "s",
                          defaultValue: .duration(90)
                      ),
                      FieldSpec(key: "repMin", name: "Reps from", kind: .number, defaultValue: .number(8)),
                      FieldSpec(key: "repMax", name: "Reps to", kind: .number, defaultValue: .number(12)),
                      FieldSpec(key: "progression", name: "Progression", kind: .text),
                      FieldSpec(key: "variantOf", name: "Variant of", kind: .classRef, target: .exercise),
                      // Swap pools and search: set on each library exercise's class object.
                      FieldSpec(key: "pattern", name: "Movement pattern", kind: .choice,
                                options: MovementPattern.allCases.map(\.rawValue)),
                      FieldSpec(key: "family", name: "Family", kind: .text),
                      FieldSpec(key: "mechanic", name: "Mechanic", kind: .choice, options: ["compound", "isolation"]),
                      FieldSpec(key: "logType", name: "Logged as", kind: .choice,
                                options: ["weightReps", "bodyweightPlus", "assisted", "repsOnly", "duration",
                                          "distanceLoad"]),
                      FieldSpec(key: "aliases", name: "Also called", kind: .text),
                  ]),
        ClassSpec(key: .strengthExercise, name: "Strength Exercise", parent: .exercise,
                  symbol: "figure.strengthtraining.traditional", colorRole: "training", traits: [.measurable], fields: [
                      FieldSpec(key: "restSeconds", name: "Rest", kind: .duration, defaultValue: .duration(120),
                                overrides: "restSeconds"),
                  ]),
        ClassSpec(key: .barbellLift, name: "Barbell Lift", parent: .strengthExercise, symbol: "dumbbell",
                  colorRole: "training", fields: [
                      FieldSpec(key: "restSeconds", name: "Rest", kind: .duration, defaultValue: .duration(180),
                                overrides: "restSeconds"),
                      FieldSpec(key: "progression", name: "Progression", kind: .text,
                                defaultValue: .text("dp(+2.5, 8, 12)"), overrides: "progression"),
                  ]),
        ClassSpec(
            key: .benchPress,
            name: "Bench Press",
            parent: .barbellLift,
            symbol: "dumbbell",
            colorRole: "training"
        ),
        ClassSpec(key: .workout, name: "Workout", parent: .activity, symbol: "figure.run", colorRole: "training",
                  traits: [.schedulable, .timed], fields: [
                      FieldSpec(key: "scheduledFor", name: "Scheduled", kind: .date),
                      FieldSpec(key: "startedAt", name: "Started", kind: .date),
                      FieldSpec(key: "endedAt", name: "Ended", kind: .date),
                      FieldSpec(key: "routine", name: "Routine", kind: .ref, target: .routine),
                  ]),
        ClassSpec(key: .exerciseEntry, name: "Exercise Entry", parent: .thing, symbol: "list.bullet",
                  colorRole: "training", fields: [
                      FieldSpec(key: "exercise", name: "Exercise", kind: .classRef, target: .exercise),
                      FieldSpec(key: "targetSets", name: "Target sets", kind: .number),
                      FieldSpec(key: "targetReps", name: "Target reps", kind: .number),
                      FieldSpec(key: "targetWeight", name: "Target weight", kind: .mass, unit: "kg"),
                      FieldSpec(key: "slot", name: "Slot", kind: .ref, target: .slot),
                      // Left without sets when the workout finished early.
                      FieldSpec(key: "skipped", name: "Skipped", kind: .bool),
                  ]),
        ClassSpec(key: .set, name: "Set", parent: .thing, symbol: "number", colorRole: "training",
                  traits: [.trackable, .measurable], fields: [
                      FieldSpec(key: "weight", name: "Weight", kind: .mass, unit: "kg"),
                      FieldSpec(key: "reps", name: "Reps", kind: .number),
                      FieldSpec(key: "rpe", name: "RPE", kind: .number),
                      FieldSpec(key: "isWarmup", name: "Warm-up", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(
                          key: "kind",
                          name: "Kind",
                          kind: .choice,
                          defaultValue: .choice(SetKind.working.rawValue),
                          options: SetKind.allCases.map(\.rawValue)
                      ),
                      FieldSpec(key: "source", name: "Logged with", kind: .choice,
                                options: SetSource.allCases.map(\.rawValue)),
                      FieldSpec(key: "loggedAt", name: "Logged", kind: .date),
                      // Beat a record when it was logged (Product 72954ac): the Day log keeps saying PR after a later
                      // session beats it. Written once by LogSet, never recomputed.
                      FieldSpec(key: "wasRecord", name: "Record", kind: .bool),
                      // The unit the weight was entered in; the weight itself is always stored in kg.
                      FieldSpec(key: "enteredUnit", name: "Entered in", kind: .choice,
                                options: MassUnit.allCases.map(\.rawValue)),
                      // Epley on effective reps: with RPE logged, 5 @ 8 counts as a 7-rep effort.
                      FieldSpec(key: "e1rm", name: "Estimated 1RM", kind: .computed, unit: "kg",
                                expression: honestE1RM),
                      FieldSpec(key: "volume", name: "Volume", kind: .computed, unit: "kg",
                                expression: "weight * reps"),
                  ]),
        // A routine (Push Day A) is an ordered list of slots, held as its children.
        ClassSpec(key: .routine, name: "Routine", parent: .thing, symbol: "list.number", colorRole: "planning",
                  fields: [
                      FieldSpec(key: "cadence", name: "Cadence", kind: .choice, defaultValue: .choice("week"),
                                options: ["day", "week", "month"]),
                      // Design 44: "2,5" for Mondays and Thursdays on a fixed-days plan (1 Sunday to 7 Saturday).
                      FieldSpec(key: "fixedWeekdays", name: "Days", kind: .text),
                      // Which kind of day it is, so its colour and muscles do not depend on the title ("Bench day").
                      // Optional: a routine of his own may have none.
                      FieldSpec(key: "kind", name: "Kind", kind: .choice, options: ["push", "pull", "legs"]),
                      // The name Tess gave the day, and the exercises it was given for: while the title is still this,
                      // Tess may rename it as the exercises change; once he types another, it never does.
                      FieldSpec(key: "autoName", name: "Name from Tess", kind: .text),
                      FieldSpec(key: "autoNameKey", name: "Named for", kind: .text),
                      // "rules" or "model": a rule-made name may be improved once by a model, nothing else may.
                      FieldSpec(key: "autoNameBy", name: "Named by", kind: .text),
                      // The last time a model was asked to improve this day's rule name (at most once a day).
                      FieldSpec(key: "autoNameTriedAt", name: "Model asked", kind: .date),
                      FieldSpec(key: "autoNameTriedKey", name: "Asked for", kind: .text),
                      // Which grove its tree stands in on the Watch; empty reads as body, as every routine is today.
                      FieldSpec(key: "grove", name: "Grove", kind: .choice, options: ["body", "mind", "home"]),
                  ]),
        // One day a routine was tended without a workout (a Watch tree held, or its steps opened): the tree reads these
        // with the finished workouts. The training ledger never reads a run.
        ClassSpec(key: .routineRun, name: "Routine Run", parent: .thing, symbol: "leaf", colorRole: "health", fields: [
            FieldSpec(key: "routine", name: "Routine", kind: .ref, target: .routine),
            FieldSpec(key: "day", name: "Day", kind: .date),
            FieldSpec(key: "doneAt", name: "Done at", kind: .date),
            FieldSpec(key: "stepsDone", name: "Steps done", kind: .number),
            FieldSpec(key: "by", name: "By", kind: .choice, defaultValue: .choice("tended"),
                      options: ["tended", "workout"]),
        ]),
        // A rest: a training rest day, or a routine paused (its tree hangs a lamp). `until` empty means open; `endedAt`
        // ends it early (tended, resumed).
        ClassSpec(key: .rest, name: "Rest", parent: .thing, symbol: "moon.zzz", colorRole: "health", fields: [
            FieldSpec(key: "scope", name: "Scope", kind: .choice, defaultValue: .choice("training"),
                      options: ["training", "routine"]),
            FieldSpec(key: "routine", name: "Routine", kind: .ref, target: .routine),
            FieldSpec(key: "from", name: "From", kind: .date),
            FieldSpec(key: "until", name: "Until", kind: .date),
            FieldSpec(key: "endedAt", name: "Ended", kind: .date),
        ]),
        ClassSpec(key: .slot, name: "Slot", parent: .thing, symbol: "rectangle.stack", colorRole: "planning",
                  fields: [
                      FieldSpec(key: "exercise", name: "Exercise", kind: .classRef, target: .exercise),
                      FieldSpec(key: "sets", name: "Sets", kind: .number, defaultValue: .number(3)),
                      FieldSpec(key: "repMin", name: "Reps from", kind: .number),
                      FieldSpec(key: "repMax", name: "Reps to", kind: .number),
                      FieldSpec(key: "targetRPE", name: "Target RPE", kind: .number),
                      FieldSpec(key: "restSeconds", name: "Rest", kind: .duration, unit: "s"),
                      FieldSpec(key: "progression", name: "Progression", kind: .text),
                  ]),
        // A plan cycles through routines in order; a missed day just moves to the next routine.
        ClassSpec(key: .plan, name: "Plan", parent: .thing, symbol: "calendar", colorRole: "planning",
                  traits: [.schedulable], fields: [
                      FieldSpec(key: "routines", name: "Routines", kind: .refList, target: .routine),
                      FieldSpec(key: "startsOn", name: "Starts", kind: .date),
                      FieldSpec(key: "weeks", name: "Weeks", kind: .number),
                      FieldSpec(key: "sessionsPerWeek", name: "Sessions a week", kind: .number),
                      // Design 44: rolls on (the default, today's behaviour) or each routine on its fixed days.
                      FieldSpec(key: "scheduleMode", name: "Schedule", kind: .choice,
                                defaultValue: .choice("rolling"), options: ["rolling", "fixed"]),
                  ]),
        ClassSpec(key: .task, name: "Task", parent: .thing, symbol: "checkmark.circle", colorRole: "planning",
                  traits: [.schedulable], fields: [
                      FieldSpec(key: "startsAt", name: "Starts", kind: .date),
                      FieldSpec(key: "length", name: "Length", kind: .duration),
                      FieldSpec(key: "due", name: "Due", kind: .date),
                      FieldSpec(key: "isDone", name: "Done", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(key: "doneAt", name: "Done at", kind: .date),
                      // Nightly roll-forward: the day it was first planned for, and how many nights it rolled.
                      FieldSpec(key: "rolledFrom", name: "First planned", kind: .date),
                      FieldSpec(key: "rolls", name: "Rolled", kind: .number),
                      // What it's for: "cook carbonara thu" links the recipe.
                      FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
                  ]),
        // Plain text with @ links; `about` holds the objects it mentions (classes via their class objects).
        ClassSpec(key: .note, name: "Note", parent: .thing, symbol: "note.text", colorRole: "neutral", fields: [
            FieldSpec(key: "body", name: "Body", kind: .text),
            FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
            // A private note (secret words, Research 134 SW4): titled "Private note" everywhere outside its page,
            // never in memory, packs or the miss log; in his backups (Q1).
            FieldSpec(key: "isPrivate", name: "Private", kind: .bool),
            // A phrase Tess couldn't read, kept to sort later (Product ruling, Oct 19): the inbox shows "To sort".
            // Added without a version bump: seeding adds a missing field to a class seeded earlier.
            FieldSpec(key: "toSort", name: "To sort", kind: .bool),
            // Pinned to the Log's Pinned window, newest pin first.
            FieldSpec(key: "pinned", name: "Pinned", kind: .bool, defaultValue: .bool(false)),
            FieldSpec(key: "pinnedAt", name: "Pinned at", kind: .date),
        ]),
        // The calendar and Health mounts (M4 slices 1 and 2): read-only mirrors, written by the system, never by him.
        ClassSpec(key: .event, name: "Event", parent: .thing, symbol: "calendar", colorRole: "neutral", fields: [
            FieldSpec(key: "start", name: "Start", kind: .date),
            FieldSpec(key: "end", name: "End", kind: .date),
            FieldSpec(key: "allDay", name: "All day", kind: .bool),
            FieldSpec(key: "calendarName", name: "Calendar", kind: .text),
            FieldSpec(key: "location", name: "Location", kind: .text),
            FieldSpec(key: "isRecurring", name: "Recurring", kind: .bool),
            FieldSpec(key: "key", name: "Key", kind: .text),
            FieldSpec(key: "busy", name: "Busy", kind: .bool),
            FieldSpec(key: "attendance", name: "Attendance", kind: .text),
            FieldSpec(key: "cancelled", name: "Cancelled", kind: .bool),
            FieldSpec(key: "eventTimeZone", name: "Time zone", kind: .text),
            FieldSpec(key: "kind", name: "Kind", kind: .text),
            FieldSpec(key: "contactId", name: "Contact", kind: .text),
            FieldSpec(key: "edited", name: "Edited", kind: .bool),
            FieldSpec(key: "countsForMe", name: "Counts for me", kind: .bool),
            FieldSpec(key: "marker", name: "Marker", kind: .bool),
            FieldSpec(key: "frozen", name: "Frozen", kind: .bool),
        ]),
        // One closed day (Shutdown, M4 slice 7): when it closed and by whom, his line for tomorrow, and the frozen Day
        // log
        // line for that day.
        ClassSpec(key: .closedDay, name: "Closed Day", parent: .thing, symbol: "moon.stars", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "date", name: "Day", kind: .date),
                      FieldSpec(key: "closedAt", name: "Closed at", kind: .date),
                      FieldSpec(key: "silent", name: "Closed by itself", kind: .bool),
                      FieldSpec(key: "tomorrowLine", name: "Tomorrow, first", kind: .text),
                      FieldSpec(key: "dayLine", name: "Day log line", kind: .text),
                  ]),
        // A Tessera notification held back while a workout, cook or focus block runs (M4 slice 4): released as one
        // digest at the next context switch. Never holds anything but the notice's own words.
        ClassSpec(key: .heldNotice, name: "Held Notice", parent: .thing, symbol: "bell.slash", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "priority", name: "Priority", kind: .text),
                      FieldSpec(key: "dueAt", name: "Due", kind: .date),
                  ]),
        // One spend (Money, M5): an amount logged like a set. Stored as typed ("12.40", "€8", "20 eur"); nothing is
        // converted and nothing is summed across currencies. Private everywhere except its own page and the Month.
        ClassSpec(key: .spend, name: "Spend", parent: .thing, symbol: "banknote", colorRole: "neutral", fields: [
            FieldSpec(key: "amount", name: "Amount", kind: .number),
            FieldSpec(key: "typed", name: "As typed", kind: .text),
            FieldSpec(key: "currency", name: "Currency", kind: .text),
            FieldSpec(key: "note", name: "Note", kind: .text),
            FieldSpec(key: "category", name: "Category", kind: .text),
            FieldSpec(key: "spentAt", name: "Spent", kind: .date),
            FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
            // Money pools: the pool it came out of. Unmarked spends read as an Unassigned bucket; nothing is rewritten.
            FieldSpec(key: "pool", name: "Pool", kind: .ref, target: .pool),
        ]),
        // A pool: where money sits, in one currency. Amounts are typed digits (Decimal, never Double). Private pools
        // read "Private pool" everywhere outside their own page.
        ClassSpec(key: .pool, name: "Pool", parent: .thing, symbol: "tray.full", colorRole: "neutral", fields: [
            FieldSpec(key: "currency", name: "Currency", kind: .text),
            FieldSpec(key: "source", name: "Source", kind: .text),
            FieldSpec(key: "poolKind", name: "Kind", kind: .choice, defaultValue: .choice("account"),
                      options: ["account", "cash", "saving"]),
            FieldSpec(key: "openingTyped", name: "Opening", kind: .text, defaultValue: .text("0.00")),
            FieldSpec(key: "isPrivate", name: "Private", kind: .bool),
            FieldSpec(key: "archivedAt", name: "Archived", kind: .date),
        ]),
        // An occasion: a trip, a move, a wedding week. Dates and links to things of any app; what it says about each
        // linked app is derived on read (OccasionFacts), never stored.
        ClassSpec(
            key: .occasion,
            name: "Occasion",
            parent: .thing,
            symbol: "calendar.badge.clock",
            colorRole: "neutral",
            fields: [
                FieldSpec(key: "start", name: "Start", kind: .date),
                FieldSpec(key: "end", name: "End", kind: .date),
                FieldSpec(key: "links", name: "Links", kind: .refList, target: .thing),
                FieldSpec(key: "away", name: "Away", kind: .bool, defaultValue: .bool(false)),
            ]
        ),
        // Money in: a paycheck, a gift, a refund. Typed digits in the pool's currency.
        ClassSpec(key: .inflow, name: "Inflow", parent: .thing, symbol: "arrow.down.circle", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "pool", name: "Pool", kind: .ref, target: .pool),
                      FieldSpec(key: "amountTyped", name: "Amount", kind: .text),
                      FieldSpec(key: "source", name: "Source", kind: .choice, defaultValue: .choice("paycheck"),
                                options: ["paycheck", "freelance", "gift", "refund", "sale"]),
                      FieldSpec(key: "at", name: "At", kind: .date),
                      FieldSpec(key: "note", name: "Note", kind: .text),
                  ]),
        // A move between two pools: what left (sent and fee, in the from currency) and what arrived (to currency).
        ClassSpec(key: .poolMove, name: "Move", parent: .thing, symbol: "arrow.left.arrow.right", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "fromPool", name: "From", kind: .ref, target: .pool),
                      FieldSpec(key: "toPool", name: "To", kind: .ref, target: .pool),
                      FieldSpec(key: "sentTyped", name: "Sent", kind: .text),
                      FieldSpec(key: "feeTyped", name: "Fee", kind: .text, defaultValue: .text("0")),
                      FieldSpec(key: "arrivedTyped", name: "Arrived", kind: .text),
                      FieldSpec(key: "at", name: "At", kind: .date),
                  ]),
        // A rate he noted or a move implied: how many of the second currency one of the first buys ("USD/INR").
        ClassSpec(key: .rateNote, name: "Rate", parent: .thing, symbol: "percent", colorRole: "neutral", fields: [
            FieldSpec(key: "pair", name: "Pair", kind: .text),
            FieldSpec(key: "valueTyped", name: "Value", kind: .text),
            FieldSpec(key: "at", name: "At", kind: .date),
            FieldSpec(key: "origin", name: "Origin", kind: .choice, defaultValue: .choice("typed"),
                      options: ["move", "typed"]),
        ]),
        // One book (Reading, M5): the page he is on is its resume point. Queued, reading, finished or dropped.
        ClassSpec(key: .book, name: "Book", parent: .thing, symbol: "book", colorRole: "neutral", fields: [
            FieldSpec(key: "author", name: "Author", kind: .text),
            FieldSpec(key: "pages", name: "Pages", kind: .number),
            FieldSpec(key: "isbn", name: "ISBN", kind: .text),
            FieldSpec(key: "state", name: "State", kind: .choice, defaultValue: .choice("queued"),
                      options: ["queued", "reading", "finished", "dropped"]),
            FieldSpec(key: "page", name: "Page", kind: .number, defaultValue: .number(0)),
            FieldSpec(key: "review", name: "Review", kind: .text),
            FieldSpec(key: "finishedAt", name: "Finished", kind: .date),
            FieldSpec(key: "isPrivate", name: "Private", kind: .bool, defaultValue: .bool(false)),
        ]),
        ClassSpec(key: .sleepNight, name: "Sleep Night", parent: .thing, symbol: "bed.double", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "date", name: "Night ending", kind: .date),
                      FieldSpec(key: "asleepMinutes", name: "Asleep", kind: .number, unit: "min"),
                      FieldSpec(key: "inBedMinutes", name: "In bed", kind: .number, unit: "min"),
                      FieldSpec(key: "awakeMinutes", name: "Awake", kind: .number, unit: "min"),
                      FieldSpec(key: "coreMinutes", name: "Core", kind: .number, unit: "min"),
                      FieldSpec(key: "deepMinutes", name: "Deep", kind: .number, unit: "min"),
                      FieldSpec(key: "remMinutes", name: "REM", kind: .number, unit: "min"),
                      FieldSpec(key: "source", name: "Source", kind: .text),
                      FieldSpec(key: "basis", name: "Basis", kind: .text),
                      FieldSpec(key: "partial", name: "Partial", kind: .bool),
                  ]),
        ClassSpec(key: .stepsDay, name: "Steps Day", parent: .thing, symbol: "figure.walk", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "date", name: "Day", kind: .date),
                      FieldSpec(key: "count", name: "Steps", kind: .number),
                  ]),
        ClassSpec(key: .restingHeartRate, name: "Resting Heart Rate", parent: .thing, symbol: "heart",
                  colorRole: "neutral", fields: [
                      FieldSpec(key: "date", name: "Day", kind: .date),
                      FieldSpec(key: "bpm", name: "Resting heart rate", kind: .number, unit: "bpm"),
                  ]),
        // One call Tess made to Claude (or a key Test): tokens, cost in nano-dollars, outcome. Written without ops;
        // never
        // holds the key. The month cap (TessUsageLedger.capNano) is a query over these.
        ClassSpec(key: .tessCall, name: "Tess Call", parent: .thing, symbol: "sparkle", colorRole: "neutral", fields: [
            FieldSpec(key: "use", name: "Use", kind: .text),
            FieldSpec(key: "model", name: "Model", kind: .text),
            FieldSpec(key: "inputTokens", name: "Input tokens", kind: .number),
            FieldSpec(key: "outputTokens", name: "Output tokens", kind: .number),
            FieldSpec(key: "costNano", name: "Cost (n$)", kind: .number),
            FieldSpec(key: "latencyMs", name: "Latency", kind: .number, unit: "ms"),
            FieldSpec(key: "outcome", name: "Outcome", kind: .text),
            FieldSpec(key: "requestID", name: "Request", kind: .text),
        ]),
        // The words of one request to Claude, exactly as sent ("Private phrase" for a walled one), beside its Tess Call
        // row and linked by requestID. On the phone only: never in a backup copy, the iCloud copy or an export, and
        // never the key. "Clear what was sent" deletes the rows; the spend stays in Tess Call (G653).
        // One turn of her conversation, on the phone only (sheet 215 rev 5): written without ops, kept seven days,
        // never in
        // any backup copy (the local one included), the iCloud copy, an export, a log or a widget. Money, health, body,
        // calorie and calendar text is never stored here, only a shown label or a placeholder.
        ClassSpec(
            key: .tessTurn,
            name: "Tess Turn",
            parent: .thing,
            symbol: "bubble.left",
            colorRole: "neutral",
            fields: [
                FieldSpec(key: "role", name: "Role", kind: .text),
                FieldSpec(key: "text", name: "Text", kind: .text),
                FieldSpec(key: "at", name: "At", kind: .date),
                FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
                FieldSpec(key: "held", name: "Held back", kind: .bool),
            ]
        ),
        ClassSpec(
            key: .tessSent,
            name: "Tess Sent",
            parent: .thing,
            symbol: "paperplane",
            colorRole: "neutral",
            fields: [
                FieldSpec(key: "use", name: "Use", kind: .text),
                FieldSpec(key: "words", name: "Words", kind: .text),
                FieldSpec(key: "requestID", name: "Request", kind: .text),
            ]
        ),
        // A Tess Sent row written in lab mode (harness H1b): the words exactly as sent, health and money included. It
        // is
        // a mirror: in no backup copy (the local one too), no export, no op.
        ClassSpec(key: .tessSentExact, name: "Tess Sent (exact)", parent: .tessSent, symbol: "paperplane.fill",
                  colorRole: "neutral", fields: []),
        // His preferences that travel with a restore (docs/research/settings-object.md): one object, a fixed id, every
        // field optional (empty = never set; readers apply defaults). Never the Claude key or the folder bookmark.
        // Him (height, birth year, sex), optional and always private: it follows a weigh-in note's rules.
        ClassSpec(key: .meProfile, name: "Me", parent: .thing, symbol: "person", colorRole: "neutral", fields: [
            FieldSpec(key: "heightCm", name: "Height", kind: .number),
            FieldSpec(key: "birthYear", name: "Birth year", kind: .number),
            FieldSpec(key: "sex", name: "Sex", kind: .text),
        ]),
        ClassSpec(key: .settings, name: "Settings", parent: .thing, symbol: "gearshape", colorRole: "neutral", fields: [
            FieldSpec(key: "lookDay", name: "Look by day", kind: .text),
            FieldSpec(key: "lookNight", name: "Look by night", kind: .text),
            FieldSpec(key: "lookPinned", name: "Pinned look", kind: .text),
            FieldSpec(key: "missesGoToClaude", name: "Misses go to Claude", kind: .bool),
            FieldSpec(key: "backupFolderName", name: "Backup folder", kind: .text),
            FieldSpec(key: "backupFolderPickedAt", name: "Folder picked", kind: .date),
            FieldSpec(key: "restWeekday", name: "Rest day", kind: .number),
            FieldSpec(key: "displayUnits", name: "Units", kind: .text),
            FieldSpec(key: "quietStartMinute", name: "Quiet hours start", kind: .number),
            FieldSpec(key: "quietEndMinute", name: "Quiet hours end", kind: .number),
            FieldSpec(key: "dayStartHour", name: "The day turns over at", kind: .number),
            FieldSpec(key: "mosaicBackdrop", name: "Mosaic scene", kind: .text),
            FieldSpec(key: "rainAlert", name: "Rain alert", kind: .bool),
            FieldSpec(key: "rainAlertLead", name: "Rain alert lead", kind: .number, unit: "min"),
            FieldSpec(key: "placeLooks", name: "Looks", kind: .text),
        ]),
        // A phrase the phone couldn't read while Claude was out of reach (tess-model-layer.md section 7): kept as said,
        // read by Claude the next time the app is open and online, and gone once read. Never commits on its own.
        ClassSpec(key: .tessWaiting, name: "Waiting for Tess", parent: .thing, symbol: "tray", colorRole: "neutral",
                  fields: [
                      FieldSpec(key: "saidAt", name: "Said", kind: .date),
                  ]),
        // One line Tess knows about him (architecture section 4, Design 49): told, noticed or recent, with the objects
        // behind it. Rides in backups; never holds health values, money, note bodies or the key (TessMemoryScreen).
        ClassSpec(key: .tessMemory, name: "Memory", parent: .thing, symbol: "brain", colorRole: "neutral", fields: [
            FieldSpec(key: "kind", name: "Kind", kind: .choice, defaultValue: .choice("told"),
                      options: ["told", "noticed", "recent", "forgotten"]),
            // Which counted rule wrote it ("training.hours", "recent.1760500800"): a forgotten rule stays quiet.
            FieldSpec(key: "rule", name: "Rule", kind: .text),
            FieldSpec(key: "area", name: "Area", kind: .choice, defaultValue: .choice("about"),
                      options: ["training", "kitchen", "people", "days", "about"]),
            FieldSpec(key: "evidence", name: "From", kind: .refList, target: .thing),
            FieldSpec(key: "lastUsedAt", name: "Used", kind: .date),
            FieldSpec(key: "pinned", name: "Pinned", kind: .bool, defaultValue: .bool(false)),
            FieldSpec(key: "isPrivate", name: "Stays on this phone", kind: .bool, defaultValue: .bool(false)),
            FieldSpec(key: "day", name: "Day", kind: .date),
        ]),
        // What Tess believes about him (docs/ai/harness/memory-context-m0-m6.md, M1). The title is the words (80
        // characters, second person; a told belief keeps his own). A belief is closed when it changes, never deleted;
        // deleting is his forget gesture. No evidence, no belief.
        ClassSpec(key: .belief, name: "Belief", parent: .thing, symbol: "sparkles", colorRole: "neutral", fields: [
            // The subject: one object, one class, or neither (him). Both set means the object.
            FieldSpec(key: "subject", name: "About", kind: .ref, target: .thing),
            FieldSpec(key: "subjectClass", name: "About class", kind: .classRef, target: .thing),
            FieldSpec(key: "key", name: "Key", kind: .choice, defaultValue: .choice("fact"),
                      options: ["usualTime", "cadence", "prefers", "avoids", "goal", "limit", "pattern", "alias",
                                "fact"]),
            // Typed JSON: {"text": ...} for words, {"hour": 18} for a usual time, and so on.
            FieldSpec(key: "value", name: "Value", kind: .text),
            FieldSpec(key: "confidence", name: "Confidence", kind: .number, defaultValue: .number(1)),
            FieldSpec(key: "source", name: "Source", kind: .choice, defaultValue: .choice("told"),
                      options: ["told", "noticed", "inferred", "learned"]),
            FieldSpec(key: "evidence", name: "From", kind: .refList, target: .thing),
            // Op ids behind it, as a JSON array of strings; an op or an object is enough.
            FieldSpec(key: "evidenceOps", name: "From ops", kind: .text),
            FieldSpec(key: "validFrom", name: "From", kind: .date),
            FieldSpec(key: "validUntil", name: "Until", kind: .date),
            FieldSpec(key: "supersedes", name: "Replaces", kind: .ref, target: .belief),
            // The last 50 times it was used, as a JSON array of ISO dates.
            FieldSpec(key: "uses", name: "Used", kind: .text),
            // How many times the record has said it again; two make an inferred belief a fact.
            FieldSpec(key: "confirmations", name: "Confirmed", kind: .number, defaultValue: .number(0)),
            FieldSpec(key: "kept", name: "Kept", kind: .bool, defaultValue: .bool(false)),
            // Which memory line it came from ("memory:<id>"), so a sync updates it in place and a forget follows it.
            FieldSpec(key: "rule", name: "Rule", kind: .text),
            FieldSpec(key: "isPrivate", name: "Stays on this phone", kind: .bool, defaultValue: .bool(false)),
        ]),
        // One of Tess's agents (docs/ai/harness/agents.md): a room's agent, with the personality it speaks to her in,
        // its notebook folder and the classes it reads. The shipped ones mirror `AgentSpec`; a new room or app spawns
        // one. The title is its room name ("Deck").
        ClassSpec(key: .tessAgent, name: "Agent", parent: .thing, symbol: "person.2", colorRole: "neutral", fields: [
            FieldSpec(key: "agentName", name: "Name", kind: .text),
            FieldSpec(key: "room", name: "Room", kind: .text),
            FieldSpec(key: "persona", name: "Personality", kind: .text),
            FieldSpec(key: "notebook", name: "Notebook", kind: .text),
            // Ids of the kinds it reads, beyond the system classes (a spawned room's own kind), as a JSON array.
            FieldSpec(key: "classIDs", name: "Kinds", kind: .text),
            FieldSpec(key: "shipped", name: "Shipped", kind: .bool, defaultValue: .bool(false)),
            FieldSpec(key: "spawnedAt", name: "Started", kind: .date),
        ]),
        // One page of an agent's own notebook (`/memories/<agent>/...`): what it keeps for itself between runs. Only
        // that agent reads or writes it. Written without ops, on the phone only, in no copy; a page written under lab
        // is never read under strict.
        ClassSpec(key: .notebookPage, name: "Notebook Page", parent: .thing, symbol: "book.closed",
                  colorRole: "neutral", fields: [
                      FieldSpec(key: "agentName", name: "Agent", kind: .text),
                      FieldSpec(key: "path", name: "Path", kind: .text),
                      FieldSpec(key: "body", name: "Body", kind: .text),
                      FieldSpec(key: "lab", name: "Written under lab", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(key: "updatedAt", name: "Updated", kind: .date),
                  ]),
        // One exchange an agent had (asked by Tess, what it reported), so the next run picks up where it left off.
        // Written without ops, kept 90 days, on the phone only, in no copy. Walled words are stored as a placeholder,
        // as her turns are.
        ClassSpec(key: .agentEpisode, name: "Agent Episode", parent: .thing, symbol: "text.bubble",
                  colorRole: "neutral", fields: [
                      FieldSpec(key: "agentName", name: "Agent", kind: .text),
                      FieldSpec(key: "at", name: "At", kind: .date),
                      FieldSpec(key: "ask", name: "Asked", kind: .text),
                      FieldSpec(key: "report", name: "Reported", kind: .text),
                      FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
                      FieldSpec(key: "lab", name: "Under lab", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(key: "held", name: "Held back", kind: .bool, defaultValue: .bool(false)),
                  ]),
        // A room Tess builds by talking (R1, docs/design/tess-built-rooms/r1-spec.md): its parts are its children. A
        // patch
        // is only create, setValue, softDelete and setSortIndex, so undo and history come free.
        ClassSpec(key: .room, name: "Room", parent: .thing, symbol: "square.dashed", colorRole: "neutral", fields: [
            FieldSpec(key: "name", name: "Name", kind: .text),
            FieldSpec(key: "color", name: "Color", kind: .choice, options: RoomKit.colorRoles),
            // JSON of LookCatalog trait values, e.g. {"density":"airy","motion":"calm"}.
            FieldSpec(key: "look", name: "Look", kind: .text),
            // JSON array of grants, e.g. ["write Plant.lastWatered"]; reads of the classes a part binds are implicit.
            FieldSpec(key: "reach", name: "Reach", kind: .text),
        ]),
        ClassSpec(key: .roomPart, name: "Room Part", parent: .thing, symbol: "square.on.square.dashed",
                  colorRole: "neutral", fields: [
                      FieldSpec(key: "kind", name: "Kind", kind: .choice, options: RoomKit.kinds.map(\.name)),
                      FieldSpec(key: "place", name: "Place", kind: .choice, options: RoomKit.places),
                      // JSON, per kind.
                      FieldSpec(key: "props", name: "Props", kind: .text),
                      // The class (and optionally a query) whose objects the part shows, resolved at render time.
                      FieldSpec(key: "bind", name: "Bind", kind: .text),
                  ]),
        // The actionable list block (groceries, packing, any list). Items are the list's children, in order.
        ClassSpec(key: .list, name: "List", parent: .thing, symbol: "list.bullet", colorRole: "planning", fields: [
            FieldSpec(key: "pinned", name: "Pinned", kind: .bool, defaultValue: .bool(false)),
            FieldSpec(key: "pinnedAt", name: "Pinned at", kind: .date),
        ]),
        ClassSpec(key: .listItem, name: "List Item", parent: .thing, symbol: "circle", colorRole: "planning",
                  fields: [
                      FieldSpec(key: "quantity", name: "Quantity", kind: .text),
                      FieldSpec(key: "isDone", name: "Done", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(key: "checkedAt", name: "Checked", kind: .date),
                      FieldSpec(key: "section", name: "Section", kind: .text),
                      FieldSpec(key: "about", name: "About", kind: .refList, target: .thing),
                      // Groceries: each recipe's (or typed line's) share, so a recipe can be taken back out.
                      FieldSpec(key: "shares", name: "Shares", kind: .text),
                  ]),
        // Something you cook with. Recipe lines and grocery items link to one; it knows its aisle and if it's a staple.
        ClassSpec(key: .ingredient, name: "Ingredient", parent: .thing, symbol: "carrot", colorRole: "health",
                  fields: [
                      FieldSpec(key: "aisle", name: "Aisle", kind: .text),
                      FieldSpec(key: "isStaple", name: "Staple", kind: .bool, defaultValue: .bool(false)),
                      // Where it lives in the Pantry (a Place) and the section inside it ("Door"); both optional.
                      FieldSpec(key: "place", name: "Place", kind: .ref, target: .place),
                      FieldSpec(key: "section", name: "Section", kind: .text),
                  ]),
        // A place in the Pantry (Fridge, Freezer, Shelf, Counter, or his own): ordered, with its sections as words.
        ClassSpec(key: .place, name: "Place", parent: .thing, symbol: "refrigerator", colorRole: "health",
                  fields: [
                      FieldSpec(key: "order", name: "Order", kind: .number),
                      FieldSpec(key: "sections", name: "Sections", kind: .text),
                  ]),
        // A recipe: its ingredient lines and steps are its children, in order. The pasted text is kept verbatim.
        ClassSpec(key: .recipe, name: "Recipe", parent: .thing, symbol: "fork.knife", colorRole: "health", fields: [
            FieldSpec(key: "serves", name: "Serves", kind: .number),
            FieldSpec(key: "activeMinutes", name: "Active", kind: .number, unit: "min"),
            FieldSpec(key: "totalMinutes", name: "Total", kind: .number, unit: "min"),
            FieldSpec(key: "sourceText", name: "Original text", kind: .text),
            // Set when the recipe page opens, without an op (kitchen week one: opened but never cooked).
            FieldSpec(key: "lastOpenedAt", name: "Last opened", kind: .date),
        ]),
        ClassSpec(key: .recipeLine, name: "Recipe Line", parent: .thing, symbol: "list.dash", colorRole: "health",
                  fields: [
                      FieldSpec(key: "amount", name: "Amount", kind: .number),
                      FieldSpec(key: "amountMax", name: "Up to", kind: .number),
                      FieldSpec(key: "unit", name: "Unit", kind: .text),
                      FieldSpec(key: "prep", name: "Prep", kind: .text),
                      FieldSpec(key: "toTaste", name: "To taste", kind: .bool),
                      FieldSpec(key: "original", name: "As written", kind: .text),
                      FieldSpec(key: "ingredient", name: "Ingredient", kind: .ref, target: .ingredient),
                      // Research 123: the pack's size rides the line ("28 oz"), so two can sizes never add up; an
                      // optional line ("for serving") stays on the recipe page and off the list.
                      FieldSpec(key: "packSize", name: "Pack size", kind: .text),
                      FieldSpec(key: "optional", name: "Optional", kind: .bool),
                  ]),
        // A step is a paragraph; its timers are minute ranges, its mentions link the ingredients it uses.
        ClassSpec(key: .recipeStep, name: "Recipe Step", parent: .thing, symbol: "text.alignleft",
                  colorRole: "health", fields: [
                      FieldSpec(key: "text", name: "Text", kind: .text),
                      FieldSpec(key: "timers", name: "Timers", kind: .text),
                      FieldSpec(key: "about", name: "Uses", kind: .refList, target: .ingredient),
                  ]),
        // One time a recipe was cooked, at a size. Stamped with cookedAt when finished.
        ClassSpec(key: .cook, name: "Cook", parent: .thing, symbol: "flame", colorRole: "health", fields: [
            FieldSpec(key: "recipe", name: "Recipe", kind: .ref, target: .recipe),
            FieldSpec(key: "serves", name: "Serves", kind: .number),
            FieldSpec(key: "cookedAt", name: "Cooked", kind: .date),
            FieldSpec(key: "note", name: "Note", kind: .text),
        ]),
        // Several recipes cooked in one session on one clock (multi-dish Cook, Part A). Status: planned, running,
        // paused, done or dropped. The plan itself is never stored: it is derived from the dishes on every read.
        ClassSpec(key: .cookSession, name: "Cook Session", parent: .thing, symbol: "flame.circle", colorRole: "health",
                  fields: [
                      FieldSpec(key: "status", name: "Status", kind: .choice, defaultValue: .choice("planned"),
                                options: ["planned", "running", "paused", "done", "dropped"]),
                      FieldSpec(key: "serveAt", name: "Serve at", kind: .date),
                      FieldSpec(key: "startedAt", name: "Started", kind: .date),
                      FieldSpec(key: "pausedAt", name: "Paused", kind: .date),
                      FieldSpec(key: "pausedTotalSec", name: "Paused total", kind: .number, defaultValue: .number(0)),
                      FieldSpec(key: "holdMaxSec", name: "Hold at most", kind: .number, defaultValue: .number(900)),
                      FieldSpec(key: "dishes", name: "Dishes", kind: .refList, target: .cookDish),
                  ]),
        // One dish of a session: a recipe at a size, its own Cook, and the numbers the plan cached for it.
        ClassSpec(key: .cookDish, name: "Cook Dish", parent: .thing, symbol: "fork.knife", colorRole: "health",
                  fields: [
                      FieldSpec(key: "session", name: "Session", kind: .ref, target: .cookSession),
                      FieldSpec(key: "recipe", name: "Recipe", kind: .ref, target: .recipe),
                      FieldSpec(key: "cook", name: "Cook", kind: .ref, target: .cook),
                      FieldSpec(key: "servings", name: "Servings", kind: .number),
                      FieldSpec(key: "order", name: "Order", kind: .number, defaultValue: .number(0)),
                      FieldSpec(key: "offsetSec", name: "Starts after", kind: .number, defaultValue: .number(0)),
                      FieldSpec(key: "heldSec", name: "Waits", kind: .number, defaultValue: .number(0)),
                      FieldSpec(key: "stepIndex", name: "Step", kind: .number, defaultValue: .number(0)),
                      FieldSpec(key: "startedAt", name: "Started", kind: .date),
                      FieldSpec(key: "isDropped", name: "Dropped", kind: .bool, defaultValue: .bool(false)),
                      FieldSpec(key: "finish", name: "Finish", kind: .choice, defaultValue: .choice("standard"),
                                options: ["standard", "holdsWarm", "last"]),
                      FieldSpec(key: "holdMinutes", name: "Holds warm", kind: .number),
                  ]),
        ClassSpec(key: .kitchenTimer, name: "Kitchen Timer", parent: .thing, symbol: "timer", colorRole: "health",
                  traits: [.timed], fields: [
                      FieldSpec(key: "startedAt", name: "Started", kind: .date),
                      FieldSpec(key: "endsAt", name: "Ends", kind: .date),
                      FieldSpec(key: "minutes", name: "Minutes", kind: .number),
                      FieldSpec(key: "isStopped", name: "Stopped", kind: .bool),
                      FieldSpec(key: "cook", name: "Cook", kind: .ref, target: .cook),
                  ]),
        ClassSpec(key: .process, name: "Process", parent: .thing, symbol: "circle.dotted", colorRole: "neutral",
                  traits: [.timed], fields: [
                      FieldSpec(key: "subject", name: "Subject", kind: .ref, target: .thing),
                      FieldSpec(key: "ownerApp", name: "App", kind: .text),
                      FieldSpec(key: "state", name: "State", kind: .choice, defaultValue: .choice("running"),
                                options: ProcessState.allCases.map(\.rawValue)),
                      FieldSpec(key: "resumePoint", name: "Resume point", kind: .text),
                      FieldSpec(key: "nextStep", name: "Next step", kind: .text),
                      FieldSpec(key: "startedAt", name: "Started", kind: .date),
                      FieldSpec(key: "lastTouchedAt", name: "Last touched", kind: .date),
                      FieldSpec(key: "waitingOn", name: "Waiting on", kind: .text),
                      FieldSpec(key: "checkAt", name: "Check at", kind: .date),
                      FieldSpec(key: "endedAt", name: "Ended", kind: .date),
                      FieldSpec(key: "endReason", name: "End reason", kind: .choice,
                                options: ProcessEndReason.allCases.map(\.rawValue)),
                  ]),
    ]

    /// Built-in classes ship as a versioned package. Bump `version` whenever a spec changes; seeding then
    /// refreshes shipped names, symbols, units, options, defaults and formulas, except what the person edited.
    public nonisolated static let package = CatalogPackage(id: "system", version: 9, name: "Tessera built-ins",
                                                           source: "Tessera", license: "Tessera")

    static let honestE1RM = "weight * (1 + (reps + 10 - coalesce(rpe, 10)) / 30)"

    /// Formulas earlier builds shipped; seeding upgrades a system field still using one of them.
    private static let retiredExpressions: [String: Set<String>] = [
        "system.set.e1rm": ["weight * (1 + reps / 30)"],
    ]

    /// Inserts any missing system classes and returns all of them by key.
    @discardableResult
    public static func seed(in context: TesseraContext) throws -> [Key: ClassDef] {
        var classes = try existing(in: context)
        for spec in specs {
            let classDef: ClassDef
            if let existing = classes[spec.key] {
                classDef = existing
                // Earlier builds called the routine class "Plan"; Plan is now its own class.
                if spec.key == .routine, existing.name == "Plan" {
                    existing.name = spec.name
                }
            } else {
                classDef = ClassDef(
                    name: spec.name,
                    parent: spec.parent.flatMap { classes[$0] },
                    symbolName: spec.symbol,
                    colorRole: spec.colorRole,
                    traits: spec.traits,
                    systemKey: spec.key.systemKey
                )
                classDef.id = stableID(spec.key.systemKey)
                context.insert(classDef)
                classes[spec.key] = classDef
            }
            let needsRefresh = classDef.packageVersion < package.version
            if needsRefresh {
                refresh(classDef, from: spec)
            }
            // Adds fields introduced in later versions to classes seeded by earlier ones.
            let present = Set((classDef.ownFields ?? []).map(\.id))
            for (index, fieldSpec) in spec.fields.enumerated() {
                let fieldID = stableID("\(spec.key.systemKey).\(fieldSpec.key)")
                if present.contains(fieldID) {
                    if needsRefresh, let field = classDef.ownFields?.first(where: { $0.id == fieldID }) {
                        refresh(field, from: fieldSpec)
                    }
                    upgradeExpression(of: fieldID, in: classDef, to: fieldSpec.expression,
                                      key: "\(spec.key.systemKey).\(fieldSpec.key)")
                    continue
                }
                let field = FieldDef(
                    name: fieldSpec.name,
                    key: fieldSpec.key,
                    kind: fieldSpec.kind,
                    sortIndex: index,
                    unit: fieldSpec.unit,
                    defaultValue: fieldSpec.defaultValue
                )
                field.id = fieldID
                field.enumOptions = fieldSpec.options
                field.expression = fieldSpec.expression
                field.targetClassID = fieldSpec.target.map { stableID($0.systemKey) }
                if let overridden = fieldSpec.overrides,
                   let parent = classDef.parent,
                   let inherited = ClassResolver.field(overridden, of: parent)
                {
                    field.overridesFieldID = inherited.id
                }
                field.owner = classDef
                context.insert(field)
            }
        }
        try context.save()
        return classes
    }

    /// Applies the package's class spec, keeping the person's edits (the overlay).
    private static func refresh(_ classDef: ClassDef, from spec: ClassSpec) {
        if !classDef.isUserModified {
            classDef.name = spec.name
            classDef.symbolName = spec.symbol
            classDef.colorRole = spec.colorRole
            classDef.traitsRaw = spec.traits.map(\.rawValue)
        }
        classDef.packageID = package.id
        classDef.packageVersion = package.version
    }

    private static func refresh(_ field: FieldDef, from spec: FieldSpec) {
        guard !field.isUserModified else { return }
        field.name = spec.name
        field.unit = spec.unit
        field.enumOptions = spec.options
        field.defaultValue = spec.defaultValue
        if spec.expression != nil {
            field.expression = spec.expression
        }
    }

    /// Moves a system formula forward, unless the person has edited it.
    private static func upgradeExpression(of fieldID: UUID, in classDef: ClassDef, to expression: String?,
                                          key: String)
    {
        guard let expression, let retired = retiredExpressions[key],
              let field = classDef.ownFields?.first(where: { $0.id == fieldID }),
              let current = field.expression, retired.contains(current) else { return }
        field.expression = expression
    }

    public static func existing(in context: TesseraContext) throws -> [Key: ClassDef] {
        let all = context.all(ClassDef.self).filter { $0.systemKey != nil }
        var result: [Key: ClassDef] = [:]
        for classDef in all {
            guard let key = Key.allCases.first(where: { $0.systemKey == classDef.systemKey }) else { continue }
            // Keep the oldest if CloudKit sync ever produced a duplicate.
            if let current = result[key], current.createdAt <= classDef.createdAt {
                continue
            }
            result[key] = classDef
        }
        return result
    }

    /// Deterministic UUID from a string (SHA-256, formatted as an RFC 4122 v5-style UUID).
    public nonisolated static func stableID(_ name: String) -> UUID {
        // The "tessera:" salt is part of every system id — never change it.
        var bytes = Array(SHA256.hash(data: Data("tessera:\(name)".utf8)).prefix(16))
        bytes[6] = (bytes[6] & 0x0F) | 0x50
        bytes[8] = (bytes[8] & 0x3F) | 0x80
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }
}
