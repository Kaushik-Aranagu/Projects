import AppIntents
import Foundation
import TesseraCore
import TesseraTess

/// What intents run against. The app registers one at launch:
/// `AppDependencyManager.shared.add(dependency: TesseraRuntime(engine: model.engine))`.
@MainActor
public final class TesseraRuntime {
    public let engine: ActionEngine
    public let tess: any TessCapturing

    public init(engine: ActionEngine, tess: (any TessCapturing)? = nil) {
        self.engine = engine
        self.tess = tess ?? TessCaptureEngine(engine: engine, model: TessModelCapture())
    }

    var store: ObjectStore {
        engine.store
    }

    var ledger: TrainingLedger {
        TrainingLedger(store: engine.store, classes: engine.classes)
    }
}

/// Every registry action, and the App Intent that exposes it (or why it's Tess-only).
/// One typed action is declared once; its App Intent is a thin wrapper that calls the same action.
public enum TesseraIntentsCatalog {
    public static let coverage: [String: String] = [
        "logSet": "LogSetIntent (via logExercise) and the Lock Screen's Done",
        "editSet": "in-app only: needs a specific set",
        "addExercise": "in-app only: needs a workout",
        "startWorkout": "StartRoutineIntent",
        "finishWorkout": "FinishWorkoutIntent",
        "logExercise": "LogSetIntent",
        "createTask": "CreateTaskIntent",
        "setTaskDone": "in-app only for now (M2 adds TaskEntity)",
        "renameObject": "in-app only",
        "deleteObject": "in-app only: destructive",
        "startProcess": "kernel-internal",
        "saveContext": "kernel-internal",
        "suspendProcess": "kernel-internal",
        "resumeProcess": "kernel-internal",
        "endProcess": "kernel-internal",
        "createRoutine": "in-app only",
        "setRoutineKind": "in-app only: the routine editor",
        "setWarmUp": "in-app only: the routine editor",
        "applyDayName": "in-app only: Tess names a training day",
        "createClass": "in-app only: the class page's New subclass",
        "syncCalendar": "system: the calendar mount on foreground",
        "syncHealth": "system: the Health mount on foreground",
        "closeDay": "in-app only: Shutdown's close",
        "createPlace": "in-app only: the Pantry",
        "renamePlace": "in-app only: the Pantry",
        "setPlaceSections": "in-app only: the Pantry",
        "movePlace": "in-app only: the Pantry",
        "removePlace": "in-app only: the Pantry",
        "setItemPlace": "in-app only: the Pantry",
        "setMeProfile": "in-app only: the You page",
        "startFocusBlock": "in-app only: the focus block",
        "focusCheckIn": "in-app only: the focus block's one check-in",
        "acceptFirstStep": "in-app only: the offered first step",
        "queueBook": "in-app only: Reading (Siri comes later)",
        "editBook": "in-app only: the book page",
        "startReading": "in-app only: Reading",
        "setBookPage": "in-app only: Reading",
        "finishBook": "in-app only: Reading",
        "setBookAside": "in-app only: Reading",
        "logSpend": "in-app only: Money (Siri comes later)",
        "editSpend": "in-app only: the spend page",
        "repeatWorkout": "in-app only: the finished workout page",
        "startCookSession": "in-app only: cooking several dishes",
        "setServeTime": "in-app only: cooking several dishes",
        "addDish": "in-app only: cooking several dishes",
        "dropDish": "in-app only: cooking several dishes",
        "advanceDishStep": "in-app only: cooking several dishes",
        "pauseCook": "in-app only: cooking several dishes",
        "resumeCook": "in-app only: cooking several dishes",
        "finishCookSession": "in-app only: cooking several dishes",
        "createPool": "in-app only: Money pools",
        "logInflow": "in-app only: Money pools",
        "reassignSpend": "in-app only: Money pools",
        "logMove": "in-app only: Money pools",
        "recordRate": "in-app only: Money pools",
        "archivePool": "in-app only: Money pools",
        "makeKind": "in-app only: Make a kind",
        "keepNoteAsObject": "in-app only: Make a kind",
        "keepAllNotes": "in-app only: Make a kind",
        "createOccasion": "in-app only: Occasions",
        "linkToOccasion": "in-app only: Occasions",
        "setAway": "in-app only: Occasions",
        "markArrived": "in-app only: Home mosaic",
        "joinTwins": "in-app only: Twins",
        "keepTwinsApart": "in-app only: Twins",
        "unlinkFromOccasion": "in-app only: Occasions",
        "logShopTotal": "in-app only: the groceries total",
        "skipShopTotal": "in-app only: the groceries total",
        "holdNotice": "system: quiet interrupts",
        "releaseHeldNotices": "system: quiet interrupts",
        "addSlot": "in-app only",
        "startRoutine": "StartRoutineIntent",
        "createPlan": "in-app only",
        "logNextSet": "LogNextSetIntent (the Lock Screen's Done)",
        "moveSlot": "in-app only: routine editor",
        "editSlot": "in-app only: routine editor",
        "removeSlot": "in-app only: routine editor",
        "swapSlotExercise": "in-app only: routine editor",
        "swapExercise": "in-app only: live mode swap",
        "rollForwardTasks": "system: nightly roll-forward",
        "startTask": "in-app only for now",
        "waitOnTask": "in-app only for now",
        "createNote": "Tell Tess (\"note: …\")",
        "editNote": "in-app only: notes app",
        "startShop": "in-app only for now (Tess \"shop\" is kitchen slice 9)",
        "finishShop": "in-app only: groceries at the shop",
        "startCooking": "in-app only for now (Tess \"cook carbonara\" is kitchen slice 9)",
        "goToCookStep": "in-app only: cook mode",
        "startKitchenTimer": "in-app only for now (Tess \"timer pasta 10\" is kitchen slice 9)",
        "stopKitchenTimer": "in-app only: cook mode",
        "finishCooking": "in-app only: cook mode",
        "addRecipeToGroceries": "in-app only: recipe page swipe",
        "removeRecipeFromGroceries": "in-app only: groceries",
        "addGroceries": "in-app only for now (Tess \"buy milk\" is kitchen slice 9)",
        "createRecipe": "in-app only for now: paste a recipe (Tess ghost is kitchen slice 9)",
        "editRecipe": "in-app only: recipe page",
        "copyList": "in-app only for now (Tess \"copy packing for lisbon\" is lists slice 8)",
        "createList": "in-app only for now (Tess \"list: …\" is kitchen slice 9)",
        "addListItem": "in-app only for now (Tess \"buy milk\" is kitchen slice 9)",
        "checkListItem": "in-app only: lists",
        "moveListItem": "in-app only: lists",
        "renameListSection": "in-app only: lists",
        "clearDoneItems": "in-app only: lists",
        "setListReusable": "in-app only: lists",
        "putAwayListRun": "in-app only: lists",
        "startListRun": "in-app only: lists (Tess \"run packing\" is Lists slice 8)",
        "finishListRun": "in-app only: lists (Tess \"done\" on a list page is Lists slice 8)",
        "setListPickOne": "in-app only: lists",
        "acceptPick": "in-app only: the pick moment (Tess \"pick from films\" is Lists slice 8)",
        "resetList": "in-app only: lists (Tess \"reset gym bag\" is Lists slice 8)",
        "finishWorkoutEarly": "in-app only: live mode's finish early",
        "moveTask": "in-app only for now (Now's \"not now\" and Later)",
        "keepPlannedTask": "in-app only: Plan my day's keep",
    ]
}
