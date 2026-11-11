import SwiftUI
import TesseraCore

/// The day stroke's colours, morning to evening, per look (themes.md, theme-contrast-fixes.md B3/B7). Dark looks
/// use the look's own `daySpectrum`. Light looks (Porcelain, Celadon) darken the sky and amber stops to reach 3:1 on
/// white; those two values are for this stroke only and never become text colours. Plein keeps its own stroke.
enum HomeStroke {
    /// Line widths, behind one place while Design tunes them (ink-line.md).
    static let day: CGFloat = 6
    static let now: CGFloat = 7.5
    static let session: CGFloat = 4
    static let carriage = CGSize(width: 40, height: 2)
    static let underAmount: CGFloat = 2

    static var gradient: Gradient {
        let look = ThemeStore.shared.paintLook
        if look.field == .plein {
            return Gradient(colors: [Color(hex: 0x9EC3D6), Color(hex: 0x8FB58A), Color(hex: 0x7A5C8E),
                                     Color(hex: 0xE0A35C), Color(hex: 0xC8452F)])
        }
        guard !look.isDark else { return Gradient(colors: Theme.Colors.daySpectrum) }
        return Gradient(colors: [Color(hex: 0x509BD5), Theme.Colors.health, Theme.Colors.context,
                                 Color(hex: 0xC58700), Theme.Colors.live])
    }
}

/// Home: three sibling pages on x, REVIEW · TODAY · TRAINING. Paged, stops at the edges, opens on TODAY and keeps
/// its page for the session. Each page fits one screen; y lives only inside bounded windows and z is a tap on any
/// name.
struct HomeSpatialView: View {
    enum Page: Int, CaseIterable, Identifiable {
        case review, today, training, kitchen

        var id: Int {
            rawValue
        }

        var title: String {
            switch self {
            case .review: "Review"
            case .today: "Today"
            case .training: "Training"
            case .kitchen: "Kitchen"
            }
        }
    }

    /// Space kept free at the bottom of every page, on top of the 56 pt RootView keeps for the soul.
    static let dockClearance: CGFloat = 48
    /// Lifts a page's foot line clear of Tess's band.
    static let footLift: CGFloat = 56

    /// Where Home was left this session (spatial memory).
    /// `-openLink` has opened its screen this launch (simulator only).
    @MainActor private static var openedLaunchLink = false
    @MainActor private static var remembered: Int = {
        #if DEBUG
            // `-homePage review|today|training|kitchen` opens Home on that page (for recordings).
            if let value = LaunchFlags.value(after: "-homePage"),
               let page = Page.allCases.first(where: { value == "\($0)" })
            {
                return page.rawValue
            }
        #endif
        return Page.today.rawValue
    }()

    @Environment(AppModel.self) private var model
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.lowPowerMotion) private var lowPower
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.scenePhase) private var scenePhase
    @State private var position = ScrollPosition(id: HomeSpatialView.remembered)
    @State private var progress = Double(HomeSpatialView.remembered)
    @State private var settled = false
    @State private var presented: HomeLink?
    @State private var pagerIdle = true
    @State private var rowGrabbed = false
    @Namespace private var linkZoom
    /// The area index (sheet 73 A): 0 closed … 1 open; the row under the finger. It opens only by a tap on the area
    /// name (a pull-down just scrolls).
    @State private var indexProgress = 0.0
    /// Whether the sheet is in the view tree at all: only while it is open or animating, so its rows are never in the
    /// accessibility tree (nor under a tap) while it is closed.
    @State private var indexMounted = false
    @State private var indexClosing = false
    @State private var picked: Int?
    @State private var arrivals = 0
    @Namespace private var areaWord
    @AccessibilityFocusState private var nameFocused: Bool

    var body: some View {
        VStack(spacing: Theme.Spacing.l) {
            AreaName(titles: Page.allCases.map(\.title), progress: progress, namespace: areaWord,
                     indexShown: indexProgress > 0.001, current: Int(progress.rounded()), flies: !calm,
                     onTap: openIndex)
                .accessibilityFocused($nameFocused)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, Theme.Spacing.gutter)
                .padding(.top, Theme.Spacing.s)
                .tesseraFeedback(.areaLands, trigger: arrivals)
                // A plain, always-visible door to Settings on every Home page (no icon, no shape), at the top right.
                .overlay(alignment: .trailing) {
                    Button { presented = .settings } label: {
                        Text("Settings")
                            .textRole(.data)
                            .foregroundStyle(Theme.Colors.inkSecondary)
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(HandleButtonStyle())
                    .padding(.trailing, Theme.Spacing.gutter - 8)
                    .padding(.top, Theme.Spacing.s)
                    .opacity(indexProgress > 0.001 ? 0 : 1)
                    .disabled(indexProgress > 0.001)
                    .accessibilityHint("Opens Settings")
                }

            ScrollView(.horizontal) {
                HStack(spacing: 0) {
                    ForEach(Page.allCases) { sibling in
                        fitted(content(for: sibling))
                            .scrollDisabled(false)
                            .containerRelativeFrame(.horizontal)
                            // The hero's parallax stays inside its own page.
                            .clipped()
                            .id(sibling.rawValue)
                    }
                }
                .scrollTargetLayout()
            }
            .scrollTargetBehavior(.paging)
            .scrollIndicators(.hidden)
            // Rows on a page only take a drag while the pager is at rest.
            .onScrollPhaseChange { _, phase in
                pagerIdle = phase == .idle
                // A swipe by hand means the pager is live, even if the restored page never applied.
                if phase != .idle {
                    settled = true
                }
            }
            // x is no longer the rail (sheet 73, his pick): a side swipe does nothing where a screen has no side page.
            // Pages still turn by position (the index); the vertical scrolls inside the pages stay live (see `fitted`).
            .scrollDisabled(true)
            // The page stays live under the index: 90%, blurred, dropped (sheet 73 A).
            .scaleEffect(calm ? 1 : 1 - 0.1 * indexProgress)
            .blur(radius: calm ? 0 : 6 * indexProgress)
            .offset(y: calm ? 0 : 40 * indexProgress)
            .environment(\.homePagerIdle, pagerIdle)
            .environment(\.homeRowGrabbed, rowGrabbed)
            .environment(\.homeRowGrab) { rowGrabbed = $0 }
            // Opens where Home was left (TODAY the first time) without a visible jump.
            .scrollPosition($position)
            .onAppear {
                position.scrollTo(id: Self.remembered)
                #if DEBUG
                    // `-openLink notes|recipes|groceries|review` opens that screen the way its Home word does.
                    // Once per launch: a rebuilt Home (the Sample data flip) doesn't open it again.
                    if !Self.openedLaunchLink, let value = LaunchFlags.value(after: "-openLink"),
                       let link = HomeLink.allCases.first(where: { value == "\($0)" })
                    {
                        Self.openedLaunchLink = true
                        Task { @MainActor in
                            try? await Task.sleep(for: .seconds(1.5))
                            presented = link
                        }
                    }
                #endif
            }
            .onScrollGeometryChange(for: Double.self) { geometry in
                geometry.containerSize.width > 0 ? geometry.contentOffset.x / geometry.containerSize.width : 0
            } action: { _, new in
                // Until the pager has landed on the remembered page, its first offsets (0, mid-layout) are not a
                // swipe: ignoring them keeps the rail's ink and carriage on TODAY from the first frame.
                guard settled || abs(new - Double(Self.remembered)) < 0.01 else { return }
                settled = true
                progress = new
            }
            .onChange(of: position.viewID(type: Int.self)) { _, new in
                if let new {
                    Self.remembered = new
                }
            }
        }
        .background {
            ThemedBackground()
            // The ground leans 2.5% (3% in a dark look) toward the area on show (sheet 86 C), one full-screen layer so
            // no seam shows under the bar or Tess's band; it follows the pager, and Today stays neutral.
            ForEach(Page.allCases) { page in
                Theme.Colors.groundTint(page.area)
                    .opacity(max(0, 1 - abs(Double(page.rawValue) - progress)))
                    .ignoresSafeArea()
            }
        }
        .toolbarVisibility(.hidden, for: .navigationBar)
        // Each of these screens keeps its own NavigationStack, so it zooms out of the word that opened it (full
        // screen, swipe down to close) instead of being pushed inside Home's stack.
        .fullScreenCover(item: $presented) { link in
            link.destination
                // Its own row above the screen, so it never sits on a title.
                .safeAreaInset(edge: .top, spacing: 0) {
                    HStack {
                        Spacer()
                        if Choice.fontIsNew() {
                            Button { presented = nil } label: {
                                Text("Close")
                                    .textRole(.label)
                                    .foregroundStyle(Theme.Colors.inkSecondary)
                                    // Never so large that it crowds the title below it.
                                    .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
                                    .frame(minWidth: 44, minHeight: 44)
                                    .contentShape(Rectangle())
                            }
                            .buttonStyle(CloseWordStyle())
                        } else {
                            Button("Close") { presented = nil }
                                .buttonStyle(InkLabelWordStyle())
                        }
                    }
                    .padding(.horizontal, Theme.Spacing.gutter - 8)
                }
                .navigationTransition(.zoom(sourceID: link, in: linkZoom))
        }
        .environment(\.openHomeLink) { presented = $0 }
        // Leaving the app with the index open closes it.
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                closeIndex()
            }
        }
        .onDisappear { closeIndex() }
        .overlay(alignment: .top) {
            if indexMounted {
                AreaIndexSheet(titles: Page.allCases.map(\.title), lines: indexLines,
                               current: Int(progress.rounded()), progress: indexProgress, picked: picked,
                               calm: calm, namespace: areaWord, onPick: pick, onClose: closeIndex)
                    .ignoresSafeArea(edges: .top)
                    .transition(.opacity)
            }
        }
        .onChange(of: indexProgress) { _, new in
            // A pull dragged back to nothing (no close animation) leaves the tree too.
            if new <= 0.001, !indexClosing {
                indexMounted = false
            }
        }
        .environment(\.homeLinkZoom, linkZoom)
        .appFan(enabled: indexProgress < 0.001, entries: fanEntries, onPick: pick, onOpenIndex: openIndex)
    }

    /// Design 137: the apps the fan lists, each with its live line (Review switch `homeV2`, Off by default).
    private var fanEntries: [AppFanEntry] {
        let lines = indexLines
        return Page.allCases.map { page in
            AppFanEntry(id: page.rawValue, title: page.title, line: lines[page.rawValue], accent: page.fanAccent)
        }
    }

    @ViewBuilder
    private func content(for sibling: Page) -> some View {
        switch sibling {
        case .review:
            ReviewPage()
                .frame(maxHeight: .infinity, alignment: .top)
                // Above Tess's band (736 to 820), never beside the soul (Design's v9 walk, R1).
                .overlay(alignment: .bottomLeading) {
                    HomeLinkLine(links: [.review, .notes]).padding(.bottom, Self.footLift)
                }
        case .today: NowPage(onShow: show)
        case .training: TrainingPage()
        // The landing, while Settings › Review › Landing pages is On (131).
        case .kitchen: if LandingSwitch.isOn {
                KitchenLandingPage()
            } else {
                KitchenPage()
            }
        }
    }

    /// One screen tall. At accessibility type sizes the page may scroll rather than truncate (reflow wins).
    @ViewBuilder
    private func fitted(_ page: some View) -> some View {
        let padded = page
            .padding(.horizontal, Theme.Spacing.gutter)
            .padding(.top, Theme.Spacing.s)
            .padding(.bottom, Self.dockClearance)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        if typeSize.isAccessibilitySize {
            // The content takes the page's width less its gutters, so a long line wraps instead of widening the
            // page and clipping both edges (AX5 on #1143's shot).
            ScrollView(.vertical) {
                page
                    .containerRelativeFrame(.horizontal, alignment: .leading) { length, _ in
                        max(length - 2 * Theme.Spacing.gutter, 0)
                    }
                    .padding(.horizontal, Theme.Spacing.gutter)
                    .padding(.top, Theme.Spacing.s)
                    .padding(.bottom, Self.dockClearance)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .scrollIndicators(.hidden)
            .scrollDisabled(false)
        } else {
            padded
        }
    }

    // MARK: The index

    /// Reduce Motion or Low Power: a 150 ms crossfade, nothing scales, blurs, drops or flies.
    private var calm: Bool {
        reduceMotion || lowPower
    }

    private var indexCurve: Animation {
        calm ? MotionPolicy.reducedMotionFade : .timingCurve(0.2, 0.9, 0.25, 1, duration: 0.46)
    }

    private var indexLines: [String] {
        let lines = AreaLines.lines(model, now: AppClock.now)
        return Page.allCases.map { lines[$0] ?? "" }
    }

    /// A tap on the name opens the index without a finger.
    private func openIndex() {
        guard !model.isCommandExpanded, !model.soul.refusesLens else { return }
        picked = nil
        // Mounted at rest first, then opened on the next turn, so the glass and the rows form as they should.
        indexMounted = true
        Task { @MainActor in
            withAnimation(indexCurve) { indexProgress = 1 }
        }
    }

    private func closeIndex() {
        guard indexProgress > 0 else { return }
        picked = nil
        nameFocused = true
        indexClosing = true
        withAnimation(indexCurve) {
            indexProgress = 0
        } completion: {
            indexClosing = false
            if indexProgress <= 0.001 {
                indexMounted = false
            }
        }
    }

    /// The chosen word flies from its row into the name's place and the page rises to full size and unblurs; Reduce
    /// Motion crossfades and the word swaps without flying.
    private func pick(_ index: Int) {
        var settle = Transaction()
        settle.disablesAnimations = true
        withTransaction(settle) {
            position.scrollTo(id: index)
            progress = Double(index)
        }
        Self.remembered = index
        picked = index
        indexClosing = true
        withAnimation(indexCurve) {
            indexProgress = 0
        } completion: {
            indexClosing = false
            indexMounted = false
            picked = nil
            arrivals += 1
            // VoiceOver lands on the opened area's front heading: with the bar word read after the page (below),
            // the first element of the new screen is that heading.
            UIAccessibility.post(notification: .screenChanged, argument: nil)
        }
    }

    private func show(_ index: Int) {
        withTesseraAnimation(.snappy(duration: 0.3), reduceMotion: reduceMotion) { position.scrollTo(id: index) }
    }

    private func show(_ sibling: Page) {
        show(sibling.rawValue)
    }
}

extension HomeSpatialView.Page {
    /// The fan's colour for the title under the finger: the app's accent, ink where it has none.
    var fanAccent: Color {
        switch self {
        case .training: Theme.Colors.Accent.training
        case .kitchen: Theme.Colors.Accent.kitchen
        case .review, .today: Theme.Colors.ink
        }
    }
}

// MARK: - Entry points

/// The screens Home reaches that are not objects: Notes, Review, Groceries, Pantry and Recipes.
enum HomeLink: Hashable, CaseIterable, Identifiable {
    case groceries, lists, pantry, recipes, pasteRecipe, notes, review, money, settings, training, today, weather

    var id: Self {
        self
    }

    var title: String {
        switch self {
        case .groceries: "Groceries"
        case .lists: "Lists"
        case .pantry: "Pantry"
        case .recipes: "Recipes"
        case .pasteRecipe: "Paste a recipe"
        case .notes: "Notes"
        case .review: "The week"
        case .money: "Money"
        case .settings: "Settings"
        case .training: "Training"
        case .today: "Today"
        case .weather: "Weather"
        }
    }

    @MainActor @ViewBuilder
    var destination: some View {
        switch self {
        case .groceries: GroceriesView()
        case .lists: ListsView()
        case .pantry: PantryView()
        // The class page of Recipe (sheet 87); a row opens its recipe.
        case .recipes: RecipesShelfView()
        case .pasteRecipe: PasteRecipeView()
        case .notes: NotesView()
        case .review: ReviewView()
        case .money: MoneyLandingView()
        case .training: ScrollView { TrainingPage().padding(.horizontal, Theme.Spacing.gutter) }
        case .today: TodayRoomView()
        case .weather: WeatherRoomView()
        // Settings as its own page from Home's top row, with Close above it (Kaushik, Oct 22: he couldn't find it).
        case .settings: NavigationStack { SettingsPage(cameFrom: "Home") }.environment(\.animateEntrance, true)
        }
    }
}

/// Kitchen: two names at hero-ish size, ink words with no shape around them; a tap zooms in.
private struct KitchenPage: View {
    @Environment(\.openHomeLink) private var open
    @Environment(\.homeLinkZoom) private var zoom

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
            Text("What we eat, and what we need.")
                .font(Theme.voice(size: 21))
                .foregroundStyle(Theme.Colors.inkSecondary)
            // Groceries, then Lists under it (lists-9-12.md, the door), then Pantry (what we have) and Recipes.
            ForEach([HomeLink.groceries, .lists, .pantry, .recipes], id: \.self) { link in
                Button { open(link) } label: {
                    // The words alone, in ink (no-dots.md).
                    Text(link.title)
                        .textRole(.title)
                        .wordFit(link.title)
                        .foregroundStyle(Theme.Colors.ink)
                        .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .modifier(LinkZoomSource(link: link, zoom: zoom))
            }
            HomeLinkLine(links: [.pasteRecipe])
            Spacer(minLength: 0)
        }
        .padding(.top, Theme.Spacing.xl)
    }
}

/// A line of entry words ("The week · Notes") at the foot of a page.
private struct HomeLinkLine: View {
    let links: [HomeLink]
    @Environment(\.openHomeLink) private var open
    @Environment(\.homeLinkZoom) private var zoom

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(links.enumerated()), id: \.offset) { index, link in
                if index > 0 {
                    Text(" · ").textRole(.body).foregroundStyle(Theme.Colors.ink.opacity(0.6))
                }
                Button { open(link) } label: {
                    Text(link.title)
                        .textRole(.bodyStrong)
                        .foregroundStyle(Theme.Colors.ink)
                        .frame(minHeight: 44)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .modifier(LinkZoomSource(link: link, zoom: zoom))
            }
        }
    }
}

/// Marks the word a Home link zooms out of, when Home provides the namespace.
private struct LinkZoomSource: ViewModifier {
    let link: HomeLink
    let zoom: Namespace.ID?

    func body(content: Content) -> some View {
        if let zoom {
            content.matchedTransitionSource(id: link, in: zoom)
        } else {
            content
        }
    }
}

extension EnvironmentValues {
    @Entry var openHomeLink: (HomeLink) -> Void = { _ in }
    @Entry var homeLinkZoom: Namespace.ID?
}

// MARK: - Sibling rail

/// Each page's hero travels at 70% of the page while paging (30% parallax). Reduce Motion: none.
private struct SiblingParallax: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.inHomeDeck) private var inDeck

    func body(content: Content) -> some View {
        // In the lens deck the pages are drawn outside the pager: no parallax.
        let amount: CGFloat = reduceMotion || inDeck ? 0 : 0.3
        // The phase is −1 … 1 as the hero's page slides out to either side, and exactly 0 at rest.
        content.scrollTransition(.interactive, axis: .horizontal) { effect, phase in
            effect.offset(x: -phase.value * amount * 390)
        }
    }
}

extension View {
    /// Apply to a sibling page's hero.
    func siblingParallax() -> some View {
        modifier(SiblingParallax())
    }
}

// MARK: - Data

/// Pure-ish helpers behind Home, kept out of the views so they can be tested.
@MainActor
enum HomeData {
    /// Upcoming items after the hero: not done, not the current task, and not already over.
    static func upcoming(_ schedule: [ScheduleItem], excluding current: ScheduleItem?, now: Date) -> [ScheduleItem] {
        schedule.filter { item in
            item.id != current?.id && !item.isDone && item.end > now && (item.isWorkout || item.start >= now)
        }
    }

    /// Yesterday, today and tomorrow, each holding the schedule items that start on that day. His day runs to
    /// 04:00 (B2): at 01:00 the today lane is still the day that's ending.
    static func lanes(_ model: AppModel, now: Date, calendar: Calendar = .current) -> [DayLane] {
        let tesseraDay = TesseraDay(calendar: calendar)
        return (-1 ... 1).compactMap { offset in
            let day = tesseraDay.day(offset, after: now)
            let items = model.schedule(around: offset == 0 ? now : day)
                .filter { tesseraDay.day(of: $0.start) == day }
            let workouts = model.ledger.finishedWorkouts(onDay: day).compactMap { summary -> ClosedRange<Date>? in
                guard let end = summary.endedAt else { return nil }
                return min(summary.startedAt ?? end, end) ... end
            }
            return DayLane(offset: offset, date: day, items: items, workouts: workouts)
        }
    }

    /// When each workout finished (or, for one still running, when it started), skipping planned ones.
    static func trainingDates(_ model: AppModel) -> [(workout: ObjectRecord, date: Date)] {
        model.objects(of: .workout).compactMap { workout in
            if case let .date(date)? = model.store.value("endedAt", of: workout) {
                return (workout, date)
            }
            if model.session?.workout.id == workout.id,
               case let .date(date)? = model.store.value("startedAt", of: workout)
            {
                return (workout, date)
            }
            return nil
        }
    }
}

// MARK: - Spoken time

/// Times and counts in words for the italic context lines: "twenty to four", "this evening at six".
enum SpokenTime {
    static func number(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        formatter.locale = Locale(identifier: "en")
        return formatter.string(from: value as NSNumber) ?? "\(value)"
    }

    /// "four", "quarter past three", "twenty to four", "seventeen minutes past three".
    static func clock(_ date: Date, calendar: Calendar = .current) -> String {
        let parts = calendar.dateComponents([.hour, .minute], from: date)
        let hour = parts.hour ?? 0, minute = parts.minute ?? 0
        func name(_ hour: Int) -> String {
            switch hour % 24 {
            case 0: "midnight"
            case 12: "noon"
            default: number((hour + 11) % 12 + 1)
            }
        }
        switch minute {
        case 0: return name(hour)
        case 15: return "quarter past \(name(hour))"
        case 30: return "half past \(name(hour))"
        case 45: return "quarter to \(name(hour + 1))"
        case let minute where minute < 30:
            return "\(number(minute))\(minute % 5 == 0 ? "" : " minutes") past \(name(hour))"
        default:
            let left = 60 - minute
            return "\(number(left))\(left % 5 == 0 ? "" : " minutes") to \(name(hour + 1))"
        }
    }

    /// "morning", "afternoon", "evening", "night".
    static func partOfDay(_ date: Date, calendar: Calendar = .current) -> String {
        switch calendar.component(.hour, from: date) {
        case 5 ..< 12: "morning"
        case 12 ..< 17: "afternoon"
        case 17 ..< 22: "evening"
        default: "night"
        }
    }

    /// "this evening, six", "tomorrow, half past seven", "Friday, nine".
    static func when(_ date: Date, now: Date, calendar: Calendar = .current) -> String {
        let atTime = ", \(clock(date, calendar: calendar))"
        // His day runs to 04:00 (B3): at 01:00 "this afternoon" means the coming afternoon only if it's still his day.
        let tesseraDay = TesseraDay(calendar: calendar)
        if tesseraDay.isSameDay(date, now) {
            let part = partOfDay(date, calendar: calendar)
            return part == "night" ? "tonight\(atTime)" : "this \(part)\(atTime)"
        }
        if tesseraDay.isSameDay(date, tesseraDay.adding(days: 1, to: now)) {
            return "tomorrow\(atTime)"
        }
        return "\(date.formatted(.dateTime.weekday(.wide)))\(atTime)"
    }
}

extension EnvironmentValues {
    /// True for the copies of Home's pages drawn in the tab lens deck.
    @Entry var inHomeDeck = false
}
