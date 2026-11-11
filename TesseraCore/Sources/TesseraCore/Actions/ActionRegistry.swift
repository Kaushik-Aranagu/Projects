import Foundation

/// Every built-in action, by name. Tess's tool list, the Claude tool schema and App Intents come from here,
/// so an action declared once is callable from all three.
public enum ActionRegistry {
    public static let descriptors: [ActionDescriptor] = [
        LogSet.descriptor, EditSet.descriptor, AddExercise.descriptor, StartWorkout.descriptor,
        FinishWorkout.descriptor, LogExercise.descriptor, CreateTask.descriptor, SetTaskDone.descriptor,
        RenameObject.descriptor, DeleteObject.descriptor, StartProcess.descriptor, SaveContext.descriptor,
        SuspendProcess.descriptor, ResumeProcess.descriptor, EndProcess.descriptor, CreateRoutine.descriptor,
        AddSlot.descriptor, StartRoutine.descriptor, CreatePlan.descriptor, LogNextSet.descriptor, MoveSlot.descriptor,
        EditSlot.descriptor, RemoveSlot.descriptor,
        SwapSlotExercise.descriptor,
        SwapEntryExercise.descriptor, MoveTask.descriptor, CreateNote.descriptor, EditNote.descriptor,
        StartShop.descriptor, FinishShop.descriptor,
        StartCooking.descriptor, GoToCookStep.descriptor, StartKitchenTimer.descriptor, StopKitchenTimer.descriptor,
        FinishCooking.descriptor,
        AddRecipeToGroceries.descriptor, RemoveRecipeFromGroceries.descriptor, AddGroceries.descriptor,
        CreateRecipe.descriptor, EditRecipe.descriptor,
        CreateList.descriptor, CopyList.descriptor, AddListItem.descriptor, CheckListItem.descriptor,
        MoveListItem.descriptor,
        RenameListSection.descriptor, ClearDoneItems.descriptor, SetListReusable.descriptor, ResetList.descriptor,
        StartListRun.descriptor, FinishListRun.descriptor, PutAwayListRun.descriptor,
        SetListPickOne.descriptor, AcceptPick.descriptor,
        FinishWorkoutEarly.descriptor,
        RollForwardTasks.descriptor, StartTask.descriptor,
        WaitOnTask.descriptor, KeepPlannedTask.descriptor, CreateClass.descriptor, SetRoutineKind.descriptor, SetWarmUp.descriptor,
        ApplyDayName.descriptor,
        SyncCalendar.descriptor, SyncHealth.descriptor, CloseDay.descriptor, CreatePlace.descriptor,
        RenamePlace.descriptor, SetPlaceSections.descriptor,
        MovePlace.descriptor, RemovePlace.descriptor, SetItemPlace.descriptor, SetMeProfile.descriptor,
        StartFocusBlock.descriptor,
        FocusCheckIn.descriptor, AcceptFirstStep.descriptor, HoldNotice.descriptor, ReleaseHeldNotices.descriptor,
        LogSpend.descriptor, EditSpend.descriptor, LogShopTotal.descriptor, SkipShopTotal.descriptor,
        QueueBook.descriptor, EditBook.descriptor, StartReading.descriptor, SetBookPage.descriptor,
        FinishBook.descriptor, SetBookAside.descriptor,
        StartCookSession.descriptor, SetServeTime.descriptor, AddDish.descriptor, DropDish.descriptor,
        AdvanceDishStep.descriptor, PauseCook.descriptor, ResumeCook.descriptor, FinishCookSession.descriptor,
        CreatePool.descriptor, LogInflow.descriptor, ReassignSpend.descriptor, LogMove.descriptor,
        RecordRate.descriptor, ArchivePool.descriptor,
        MakeKind.descriptor, KeepNoteAsObject.descriptor, KeepAllNotes.descriptor,
        CreateOccasion.descriptor, LinkToOccasion.descriptor, UnlinkFromOccasion.descriptor, SetAway.descriptor,
        MarkArrived.descriptor, RepeatWorkout.descriptor, JoinTwins.descriptor, KeepTwinsApart.descriptor,
    ]

    /// The actions Tess and the model's tool list may call; unoffered ones stay in `descriptors` for decoding.
    public static var offered: [ActionDescriptor] {
        descriptors.filter(\.isOfferedToModel)
    }

    /// The compact catalog for the on-device model's instructions, one action per line.
    public static var catalog: String {
        offered.map(\.catalogLine).joined(separator: "\n")
    }

    /// Tool definitions as JSON, ready for a Claude request's `tools` array.
    public static func toolDefinitionsJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(offered.map(\.toolDefinition))
    }

    /// Decodes a tool call's JSON arguments into a typed action. Dates are ISO 8601.
    public static func decode(name: String, arguments: Data) throws -> any EngineAction {
        guard let decoder = decoders[name] else { throw ActionError.unknownAction(name) }
        return try decoder(arguments)
    }

    private static let decoders: [String: @Sendable (Data) throws -> any EngineAction] = [
        MoveTask.descriptor.name: { try decode(MoveTask.self, MoveTask.descriptor.name, $0) },
        KeepPlannedTask.descriptor.name: { try decode(KeepPlannedTask.self, KeepPlannedTask.descriptor.name, $0) },
        CreateNote.descriptor.name: { try decode(CreateNote.self, CreateNote.descriptor.name, $0) },
        EditNote.descriptor.name: { try decode(EditNote.self, EditNote.descriptor.name, $0) },
        StartShop.descriptor.name: { try decode(StartShop.self, StartShop.descriptor.name, $0) },
        FinishShop.descriptor.name: { try decode(FinishShop.self, FinishShop.descriptor.name, $0) },
        StartCooking.descriptor.name: { try decode(StartCooking.self, StartCooking.descriptor.name, $0) },
        GoToCookStep.descriptor.name: { try decode(GoToCookStep.self, GoToCookStep.descriptor.name, $0) },
        StartKitchenTimer.descriptor
            .name: { try decode(StartKitchenTimer.self, StartKitchenTimer.descriptor.name, $0) },
        StopKitchenTimer.descriptor.name: { try decode(StopKitchenTimer.self, StopKitchenTimer.descriptor.name, $0) },
        FinishCooking.descriptor.name: { try decode(FinishCooking.self, FinishCooking.descriptor.name, $0) },
        AddRecipeToGroceries.descriptor.name: { try decode(
            AddRecipeToGroceries.self,
            AddRecipeToGroceries.descriptor.name,
            $0
        ) },
        RemoveRecipeFromGroceries.descriptor.name: { try decode(
            RemoveRecipeFromGroceries.self,
            RemoveRecipeFromGroceries.descriptor.name,
            $0
        ) },
        AddGroceries.descriptor.name: { try decode(AddGroceries.self, AddGroceries.descriptor.name, $0) },
        CreateRecipe.descriptor.name: { try decode(CreateRecipe.self, CreateRecipe.descriptor.name, $0) },
        EditRecipe.descriptor.name: { try decode(EditRecipe.self, EditRecipe.descriptor.name, $0) },
        CreateList.descriptor.name: { try decode(CreateList.self, CreateList.descriptor.name, $0) },
        CopyList.descriptor.name: { try decode(CopyList.self, CopyList.descriptor.name, $0) },
        AddListItem.descriptor.name: { try decode(AddListItem.self, AddListItem.descriptor.name, $0) },
        CheckListItem.descriptor.name: { try decode(CheckListItem.self, CheckListItem.descriptor.name, $0) },
        MoveListItem.descriptor.name: { try decode(MoveListItem.self, MoveListItem.descriptor.name, $0) },
        RenameListSection.descriptor
            .name: { try decode(RenameListSection.self, RenameListSection.descriptor.name, $0) },
        ClearDoneItems.descriptor.name: { try decode(ClearDoneItems.self, ClearDoneItems.descriptor.name, $0) },
        SetListReusable.descriptor.name: { try decode(SetListReusable.self, SetListReusable.descriptor.name, $0) },
        ResetList.descriptor.name: { try decode(ResetList.self, ResetList.descriptor.name, $0) },
        StartListRun.descriptor.name: { try decode(StartListRun.self, StartListRun.descriptor.name, $0) },
        FinishListRun.descriptor.name: { try decode(FinishListRun.self, FinishListRun.descriptor.name, $0) },
        PutAwayListRun.descriptor.name: { try decode(PutAwayListRun.self, PutAwayListRun.descriptor.name, $0) },
        SetListPickOne.descriptor.name: { try decode(SetListPickOne.self, SetListPickOne.descriptor.name, $0) },
        AcceptPick.descriptor.name: { try decode(AcceptPick.self, AcceptPick.descriptor.name, $0) },
        FinishWorkoutEarly.descriptor.name: {
            try decode(FinishWorkoutEarly.self, FinishWorkoutEarly.descriptor.name, $0)
        },
        StartTask.descriptor.name: { try decode(StartTask.self, StartTask.descriptor.name, $0) },
        WaitOnTask.descriptor.name: { try decode(WaitOnTask.self, WaitOnTask.descriptor.name, $0) },
        RollForwardTasks.descriptor.name: { try decode(RollForwardTasks.self, RollForwardTasks.descriptor.name, $0) },
        LogSet.descriptor.name: { try decode(LogSet.self, LogSet.descriptor.name, $0) },
        EditSet.descriptor.name: { try decode(EditSet.self, EditSet.descriptor.name, $0) },
        AddExercise.descriptor.name: { try decode(AddExercise.self, AddExercise.descriptor.name, $0) },
        StartWorkout.descriptor.name: { try decode(StartWorkout.self, StartWorkout.descriptor.name, $0) },
        FinishWorkout.descriptor.name: { try decode(FinishWorkout.self, FinishWorkout.descriptor.name, $0) },
        LogExercise.descriptor.name: { try decode(LogExercise.self, LogExercise.descriptor.name, $0) },
        CreateTask.descriptor.name: { try decode(CreateTask.self, CreateTask.descriptor.name, $0) },
        SetTaskDone.descriptor.name: { try decode(SetTaskDone.self, SetTaskDone.descriptor.name, $0) },
        RenameObject.descriptor.name: { try decode(RenameObject.self, RenameObject.descriptor.name, $0) },
        DeleteObject.descriptor.name: { try decode(DeleteObject.self, DeleteObject.descriptor.name, $0) },
        StartProcess.descriptor.name: { try decode(StartProcess.self, StartProcess.descriptor.name, $0) },
        SaveContext.descriptor.name: { try decode(SaveContext.self, SaveContext.descriptor.name, $0) },
        SuspendProcess.descriptor.name: { try decode(SuspendProcess.self, SuspendProcess.descriptor.name, $0) },
        ResumeProcess.descriptor.name: { try decode(ResumeProcess.self, ResumeProcess.descriptor.name, $0) },
        EndProcess.descriptor.name: { try decode(EndProcess.self, EndProcess.descriptor.name, $0) },
        CreateRoutine.descriptor.name: { try decode(CreateRoutine.self, CreateRoutine.descriptor.name, $0) },
        SetRoutineKind.descriptor.name: { try decode(SetRoutineKind.self, SetRoutineKind.descriptor.name, $0) },
        SetWarmUp.descriptor.name: { try decode(SetWarmUp.self, SetWarmUp.descriptor.name, $0) },
        ApplyDayName.descriptor.name: { try decode(ApplyDayName.self, ApplyDayName.descriptor.name, $0) },
        CreateClass.descriptor.name: { try decode(CreateClass.self, CreateClass.descriptor.name, $0) },
        SyncCalendar.descriptor.name: { try decode(SyncCalendar.self, SyncCalendar.descriptor.name, $0) },
        CloseDay.descriptor.name: { try decode(CloseDay.self, CloseDay.descriptor.name, $0) },
        CreatePlace.descriptor.name: { try decode(CreatePlace.self, CreatePlace.descriptor.name, $0) },
        RenamePlace.descriptor.name: { try decode(RenamePlace.self, RenamePlace.descriptor.name, $0) },
        SetPlaceSections.descriptor.name: { try decode(SetPlaceSections.self, SetPlaceSections.descriptor.name, $0) },
        MovePlace.descriptor.name: { try decode(MovePlace.self, MovePlace.descriptor.name, $0) },
        RemovePlace.descriptor.name: { try decode(RemovePlace.self, RemovePlace.descriptor.name, $0) },
        SetItemPlace.descriptor.name: { try decode(SetItemPlace.self, SetItemPlace.descriptor.name, $0) },
        SetMeProfile.descriptor.name: { try decode(SetMeProfile.self, SetMeProfile.descriptor.name, $0) },
        StartFocusBlock.descriptor.name: { try decode(StartFocusBlock.self, StartFocusBlock.descriptor.name, $0) },
        FocusCheckIn.descriptor.name: { try decode(FocusCheckIn.self, FocusCheckIn.descriptor.name, $0) },
        AcceptFirstStep.descriptor.name: { try decode(AcceptFirstStep.self, AcceptFirstStep.descriptor.name, $0) },
        HoldNotice.descriptor.name: { try decode(HoldNotice.self, HoldNotice.descriptor.name, $0) },
        ReleaseHeldNotices.descriptor
            .name: { try decode(ReleaseHeldNotices.self, ReleaseHeldNotices.descriptor.name, $0) },
        LogSpend.descriptor.name: { try decode(LogSpend.self, LogSpend.descriptor.name, $0) },
        EditSpend.descriptor.name: { try decode(EditSpend.self, EditSpend.descriptor.name, $0) },
        LogShopTotal.descriptor.name: { try decode(LogShopTotal.self, LogShopTotal.descriptor.name, $0) },
        SkipShopTotal.descriptor.name: { try decode(SkipShopTotal.self, SkipShopTotal.descriptor.name, $0) },
        StartCookSession.descriptor.name: { try decode(StartCookSession.self, StartCookSession.descriptor.name, $0) },
        SetServeTime.descriptor.name: { try decode(SetServeTime.self, SetServeTime.descriptor.name, $0) },
        AddDish.descriptor.name: { try decode(AddDish.self, AddDish.descriptor.name, $0) },
        DropDish.descriptor.name: { try decode(DropDish.self, DropDish.descriptor.name, $0) },
        AdvanceDishStep.descriptor.name: { try decode(AdvanceDishStep.self, AdvanceDishStep.descriptor.name, $0) },
        PauseCook.descriptor.name: { try decode(PauseCook.self, PauseCook.descriptor.name, $0) },
        ResumeCook.descriptor.name: { try decode(ResumeCook.self, ResumeCook.descriptor.name, $0) },
        FinishCookSession.descriptor.name: { try decode(FinishCookSession.self, FinishCookSession.descriptor.name, $0) },
        CreatePool.descriptor.name: { try decode(CreatePool.self, CreatePool.descriptor.name, $0) },
        LogInflow.descriptor.name: { try decode(LogInflow.self, LogInflow.descriptor.name, $0) },
        ReassignSpend.descriptor.name: { try decode(ReassignSpend.self, ReassignSpend.descriptor.name, $0) },
        LogMove.descriptor.name: { try decode(LogMove.self, LogMove.descriptor.name, $0) },
        RecordRate.descriptor.name: { try decode(RecordRate.self, RecordRate.descriptor.name, $0) },
        ArchivePool.descriptor.name: { try decode(ArchivePool.self, ArchivePool.descriptor.name, $0) },
        MakeKind.descriptor.name: { try decode(MakeKind.self, MakeKind.descriptor.name, $0) },
        KeepNoteAsObject.descriptor.name: { try decode(KeepNoteAsObject.self, KeepNoteAsObject.descriptor.name, $0) },
        KeepAllNotes.descriptor.name: { try decode(KeepAllNotes.self, KeepAllNotes.descriptor.name, $0) },
        CreateOccasion.descriptor.name: { try decode(CreateOccasion.self, CreateOccasion.descriptor.name, $0) },
        SetAway.descriptor.name: { try decode(SetAway.self, SetAway.descriptor.name, $0) },
        LinkToOccasion.descriptor.name: { try decode(LinkToOccasion.self, LinkToOccasion.descriptor.name, $0) },
        UnlinkFromOccasion.descriptor.name: { try decode(UnlinkFromOccasion.self, UnlinkFromOccasion.descriptor.name, $0) },
        JoinTwins.descriptor.name: { try decode(JoinTwins.self, JoinTwins.descriptor.name, $0) },
        KeepTwinsApart.descriptor.name: { try decode(KeepTwinsApart.self, KeepTwinsApart.descriptor.name, $0) },
        MarkArrived.descriptor.name: { try decode(MarkArrived.self, MarkArrived.descriptor.name, $0) },
        RepeatWorkout.descriptor.name: { try decode(RepeatWorkout.self, RepeatWorkout.descriptor.name, $0) },
        QueueBook.descriptor.name: { try decode(QueueBook.self, QueueBook.descriptor.name, $0) },
        EditBook.descriptor.name: { try decode(EditBook.self, EditBook.descriptor.name, $0) },
        StartReading.descriptor.name: { try decode(StartReading.self, StartReading.descriptor.name, $0) },
        SetBookPage.descriptor.name: { try decode(SetBookPage.self, SetBookPage.descriptor.name, $0) },
        FinishBook.descriptor.name: { try decode(FinishBook.self, FinishBook.descriptor.name, $0) },
        SetBookAside.descriptor.name: { try decode(SetBookAside.self, SetBookAside.descriptor.name, $0) },
        SyncHealth.descriptor.name: { try decode(SyncHealth.self, SyncHealth.descriptor.name, $0) },
        AddSlot.descriptor.name: { try decode(AddSlot.self, AddSlot.descriptor.name, $0) },
        StartRoutine.descriptor.name: { try decode(StartRoutine.self, StartRoutine.descriptor.name, $0) },
        CreatePlan.descriptor.name: { try decode(CreatePlan.self, CreatePlan.descriptor.name, $0) },
        AddRoutineToPlan.descriptor.name: { try decode(AddRoutineToPlan.self, AddRoutineToPlan.descriptor.name, $0) },
        MoveRoutineInPlan.descriptor.name: {
            try decode(MoveRoutineInPlan.self, MoveRoutineInPlan.descriptor.name, $0)
        },
        DropRoutineFromPlan.descriptor.name: {
            try decode(DropRoutineFromPlan.self, DropRoutineFromPlan.descriptor.name, $0)
        },
        SetScheduleMode.descriptor.name: { try decode(SetScheduleMode.self, SetScheduleMode.descriptor.name, $0) },
        SetRoutineWeekdays.descriptor.name: {
            try decode(SetRoutineWeekdays.self, SetRoutineWeekdays.descriptor.name, $0)
        },
        RememberFact.descriptor.name: { try decode(RememberFact.self, RememberFact.descriptor.name, $0) },
        ForgetMemory.descriptor.name: { try decode(ForgetMemory.self, ForgetMemory.descriptor.name, $0) },
        SetPinned.descriptor.name: { try decode(SetPinned.self, SetPinned.descriptor.name, $0) },
        SetRainAlert.descriptor.name: { try decode(SetRainAlert.self, SetRainAlert.descriptor.name, $0) },
        TakeRest.descriptor.name: { try decode(TakeRest.self, TakeRest.descriptor.name, $0) },
        ResumeRoutine.descriptor.name: { try decode(ResumeRoutine.self, ResumeRoutine.descriptor.name, $0) },
        TendRoutine.descriptor.name: { try decode(TendRoutine.self, TendRoutine.descriptor.name, $0) },
        PlantRoutine.descriptor.name: { try decode(PlantRoutine.self, PlantRoutine.descriptor.name, $0) },
        ForgetAllMemory.descriptor.name: { try decode(ForgetAllMemory.self, ForgetAllMemory.descriptor.name, $0) },
        RecordBelief.descriptor.name: { try decode(RecordBelief.self, RecordBelief.descriptor.name, $0) },
        ReviseBelief.descriptor.name: { try decode(ReviseBelief.self, ReviseBelief.descriptor.name, $0) },
        ConfirmBelief.descriptor.name: { try decode(ConfirmBelief.self, ConfirmBelief.descriptor.name, $0) },
        MergeBelief.descriptor.name: { try decode(MergeBelief.self, MergeBelief.descriptor.name, $0) },
        ForgetBelief.descriptor.name: { try decode(ForgetBelief.self, ForgetBelief.descriptor.name, $0) },
        LogNextSet.descriptor.name: { try decode(LogNextSet.self, LogNextSet.descriptor.name, $0) },
        MoveSlot.descriptor.name: { try decode(MoveSlot.self, MoveSlot.descriptor.name, $0) },
        EditSlot.descriptor.name: { try decode(EditSlot.self, EditSlot.descriptor.name, $0) },
        RemoveSlot.descriptor.name: { try decode(RemoveSlot.self, RemoveSlot.descriptor.name, $0) },
        SwapSlotExercise.descriptor.name: { try decode(SwapSlotExercise.self, SwapSlotExercise.descriptor.name, $0) },
        SwapEntryExercise.descriptor
            .name: { try decode(SwapEntryExercise.self, SwapEntryExercise.descriptor.name, $0) },
    ]

    private static func decode<A: EngineAction>(_ type: A.Type, _ name: String, _ arguments: Data) throws -> A {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(type, from: arguments)
        } catch let DecodingError.keyNotFound(key, _) {
            throw ActionError.invalidArguments(action: name, reason: "missing \(key.stringValue)")
        } catch let DecodingError.typeMismatch(_, context) {
            throw ActionError.invalidArguments(action: name, reason: "wrong type for \(path(context))")
        } catch let DecodingError.valueNotFound(_, context) {
            throw ActionError.invalidArguments(action: name, reason: "missing value for \(path(context))")
        } catch {
            throw ActionError.invalidArguments(action: name, reason: "unreadable arguments")
        }
    }

    private static func path(_ context: DecodingError.Context) -> String {
        context.codingPath.map(\.stringValue).joined(separator: ".")
    }
}
