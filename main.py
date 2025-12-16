from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import customtkinter as ctk


@dataclass
class appSettings:
    appearanceMode: str = "dark"
    uiTheme: str = "blue"
    plotTheme: str = ""


def getAppDirPath() -> Path:
    homeDirPath: Path = Path.home()
    docDirPath: Path = homeDirPath / "Documents"
    appDirPath: Path = docDirPath / "IGRS"
    appDirPath.mkdir(parents=True, exist_ok=True)
    return appDirPath


def getSettingsFilePath() -> Path:
    return getAppDirPath() / "settings.json"


def getFinalListCsvPath() -> Path:
    return getAppDirPath() / "final_list.csv"


def getThemeDirPath() -> Path:
    themeDirPath: Path = getAppDirPath() / "theme"
    themeDirPath.mkdir(parents=True, exist_ok=True)
    return themeDirPath


def getUiThemeDirPath() -> Path:
    uiThemeDirPath: Path = getThemeDirPath() / "ui"
    uiThemeDirPath.mkdir(parents=True, exist_ok=True)
    return uiThemeDirPath


def getPlotThemeDirPath() -> Path:
    plotThemeDirPath: Path = getThemeDirPath() / "plot"
    plotThemeDirPath.mkdir(parents=True, exist_ok=True)
    return plotThemeDirPath


def getCacheDirPath() -> Path:
    cacheDirPath: Path = getAppDirPath() / "cache"
    cacheDirPath.mkdir(parents=True, exist_ok=True)
    return cacheDirPath


def getEmbeddingCacheMetaPath() -> Path:
    return getCacheDirPath() / "gift_embeddings_meta.json"


def getEmbeddingCacheArrayPath() -> Path:
    return getCacheDirPath() / "gift_embeddings.npy"


def loadAppSettingsFromDisk() -> appSettings:
    settingsPath: Path = getSettingsFilePath()
    if not settingsPath.exists():
        return appSettings()

    try:
        with settingsPath.open("r", encoding="utf-8") as settingsFile:
            rawData: Dict[str, Any] = json.load(settingsFile)

        return appSettings(
            appearanceMode=str(rawData.get("appearanceMode", "dark")),
            uiTheme=str(rawData.get("uiTheme", "blue")),
            plotTheme=str(rawData.get("plotTheme", "")),
        )
    except Exception:
        traceback.print_exc()
        return appSettings()


def saveAppSettingsToDisk(settings: appSettings) -> None:
    settingsFilePath: Path = getSettingsFilePath()
    with settingsFilePath.open("w", encoding="utf-8") as settingsFile:
        json.dump(asdict(settings), settingsFile, indent=4)


columnDef = Tuple[str, str, int]


class virtualizedTable(ctk.CTkFrame):
    def __init__(self, master: Any, *,
        columns: Sequence[columnDef], rowHeight: int = 34,
        overscan: int = 2, maxPool: int = 30, rowPady: int = 2) -> None:
        super().__init__(master)
        self.columns: List[columnDef] = list(columns)
        self.rowHeight: int = rowHeight
        self.overscan: int = overscan
        self.maxPool: int = maxPool
        self.rowPady: int = rowPady

        self.itemsAll: List[Dict[str, str]] = []
        self.itemsView: List[Dict[str, str]] = []
        self.startIndex: int = 0

        self.poolRows: List[Dict[str, Any]] = []
        self.poolSize: int = 0

        self.poolContainer: ctk.CTkFrame
        self.scrollbar: ctk.CTkScrollbar

        self._buildUi()

    def _buildUi(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header: ctk.CTkFrame = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 6))

        for colIdx, (_key, title, weight) in enumerate(self.columns):
            header.grid_columnconfigure(colIdx, weight=weight, uniform="cols")
            label: ctk.CTkLabel = ctk.CTkLabel(header, text=title, anchor="w")
            label.grid(row=0, column=colIdx, sticky="ew", padx=(8, 4) if colIdx < len(self.columns) - 1 else (8, 8))

        self.poolContainer = ctk.CTkFrame(self)
        self.poolContainer.grid(row=1, column=0, sticky="nsew")
        self.poolContainer.grid_columnconfigure(0, weight=1)

        self.scrollbar = ctk.CTkScrollbar(self, command=self.onScrollbar)
        self.scrollbar.grid(row=1, column=1, sticky="ns", padx=(6, 4), pady=(2, 2))

        self.poolContainer.bind("<Configure>", lambda _e: self.after(10, self.rebuildPool), add=True)

    def setItems(self, items: List[Dict[str, str]]) -> None:
        self.itemsAll = list(items)
        self.itemsView = list(items)
        self.startIndex = 0
        self.rebuildPool()
        self.rebindRows()

    def visibleRowCount(self) -> int:
        return max(1, self.poolSize - self.overscan)

    def rowStride(self) -> int:
        return self.rowHeight + (self.rowPady * 2)

    def clampStartIndex(self) -> None:
        totalCount: int = len(self.itemsView)
        if totalCount <= 0:
            self.startIndex = 0
            return

        visibleCount: int = self.visibleRowCount()
        maxStartIndex: int = max(0, totalCount - visibleCount)

        if self.startIndex < 0:
            self.startIndex = 0
        elif self.startIndex > maxStartIndex:
            self.startIndex = maxStartIndex

    def rebuildPool(self) -> None:
        viewportHeight: int = max(1, int(self.poolContainer.winfo_height()))
        rowsFit: int = max(1, viewportHeight // self.rowStride())
        desiredPoolSize: int = min(rowsFit + self.overscan, self.maxPool)

        if desiredPoolSize == self.poolSize and self.poolRows:
            self.rebindRows()
            self.updateThumb()
            return

        for poolRow in self.poolRows:
            try:
                rowFrame: ctk.CTkFrame = poolRow["frame"]
                rowFrame.destroy()
            except Exception:
                pass
        self.poolRows.clear()

        for poolRowIndex in range(desiredPoolSize):
            rowFrame = ctk.CTkFrame(self.poolContainer, height=self.rowHeight)
            rowFrame.grid(row=poolRowIndex, column=0, sticky="ew", padx=4, pady=self.rowPady)
            rowFrame.grid_propagate(False)

            for colIndex, (_key, _title, weight) in enumerate(self.columns):
                rowFrame.grid_columnconfigure(colIndex, weight=weight, uniform="cols")

            varsByKey: Dict[str, ctk.StringVar] = {}
            for colIndex, (key, _title, _weight) in enumerate(self.columns):
                var: ctk.StringVar = ctk.StringVar(value="")
                varsByKey[key] = var
                label = ctk.CTkLabel(rowFrame, textvariable=var, anchor="w")
                label.grid(row=0, column=colIndex, sticky="ew", padx=(8, 4) if colIndex < len(self.columns) - 1 else (8, 8), pady=4)

            self.poolRows.append({"frame": rowFrame, "vars": varsByKey})

        self.poolSize = desiredPoolSize
        self.rebindRows()
        self.updateThumb()

    def rebindRows(self) -> None:
        if not self.poolRows:
            return

        self.clampStartIndex()

        totalCount: int = len(self.itemsView)
        for poolRowOffset, poolRow in enumerate(self.poolRows):
            dataIndex: int = self.startIndex + poolRowOffset
            varsByKey: Dict[str, ctk.StringVar] = poolRow["vars"]

            if dataIndex >= totalCount:
                for key, _title, _weight in self.columns:
                    varsByKey[key].set("")
                continue

            item: Dict[str, str] = self.itemsView[dataIndex]
            for key, _title, _weight in self.columns:
                varsByKey[key].set(item.get(key, ""))

        self.updateThumb()

    def updateThumb(self) -> None:
        totalCount: int = len(self.itemsView)
        visibleCount: int = self.visibleRowCount()

        if totalCount <= visibleCount:
            try:
                self.scrollbar.set(0.0, 1.0)
            except Exception:
                pass
            return

        firstFrac: float = self.startIndex / totalCount
        lastFrac: float = (self.startIndex + visibleCount) / totalCount
        try:
            self.scrollbar.set(firstFrac, lastFrac)
        except Exception:
            pass

    def onScrollbar(self, *args: Any) -> None:
        if not args:
            return

        op: Any = args[0]
        if op == "moveto":
            try:
                frac: float = float(args[1])
            except Exception:
                return

            totalCount: int = len(self.itemsView)
            visibleCount: int = self.visibleRowCount()
            maxStartIndex: int = max(0, totalCount - visibleCount)

            self.startIndex = int(round(frac * maxStartIndex))
            self.rebindRows()

        elif op == "scroll":
            try:
                count: int = int(args[1])
            except Exception:
                return

            what: Any = args[2] if len(args) > 2 else "units"
            step: int = self.visibleRowCount() if what == "pages" else 3

            self.startIndex += count * step
            self.rebindRows()

    def onMouseWheel(self, event: Any) -> None:
        delta: int = 0
        try:
            if getattr(event, "delta", 0):
                delta = 1 if event.delta > 0 else -1
            elif getattr(event, "num", None) == 4:
                delta = 1
            elif getattr(event, "num", None) == 5:
                delta = -1
        except Exception:
            delta = 0

        if delta != 0:
            self.startIndex -= delta * 3
            self.rebindRows()

    def scrollPages(self, pages: int) -> None:
        self.startIndex += pages * self.visibleRowCount()
        self.rebindRows()

    def scrollHome(self) -> None:
        self.startIndex = 0
        self.rebindRows()

    def scrollEnd(self) -> None:
        totalCount: int = len(self.itemsView)
        self.startIndex = max(0, totalCount - self.visibleRowCount())
        self.rebindRows()


class giftRecommender:
    def __init__(self) -> None:
        self.model: Any = None
        self.np: Any = None

        self.giftRows: List[Dict[str, str]] = []
        self.giftTexts: List[str] = []
        self.giftEmbeddings: Any = None

        self.modelName: str = "sentence-transformers/all-MiniLM-L6-v2"

    def ensureReady(self) -> None:
        if self.model is not None and self.np is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer #type: ignore
            import numpy as np #type: ignore
        except Exception as exc:
            raise RuntimeError("Install deps: pip install sentence-transformers numpy") from exc

        self.np = np
        self.model = SentenceTransformer(self.modelName)

    def _giftText(self, giftRow: Dict[str, str]) -> str:
        giftName: str = giftRow.get("gift", "")
        category: str = giftRow.get("category", "")
        ageLimit: str = giftRow.get("ageLimit", "")
        return f"gift: {giftName}. category: {category}. ageLimit: {ageLimit}"

    def _buildCacheKey(self, giftTexts: Sequence[str]) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.modelName.encode("utf-8"))
        hasher.update(b"\n")
        for giftText in giftTexts:
            hasher.update(giftText.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _tryLoadEmbeddingsFromCache(self, cacheKey: str) -> Optional[Any]:
        metaPath: Path = getEmbeddingCacheMetaPath()
        arrayPath: Path = getEmbeddingCacheArrayPath()

        if not metaPath.exists() or not arrayPath.exists():
            return None

        try:
            with metaPath.open("r", encoding="utf-8") as metaFile:
                meta: Dict[str, Any] = json.load(metaFile)

            if str(meta.get("cacheKey", "")) != cacheKey:
                return None
            if str(meta.get("modelName", "")) != self.modelName:
                return None

            try:
                import numpy as np #type: ignore
            except Exception:
                return None

            loadedArray: Any = np.load(str(arrayPath), allow_pickle=False)
            return loadedArray
        except Exception:
            return None

    def _saveEmbeddingsToCache(self, cacheKey: str, embeddings: Any) -> None:
        metaPath: Path = getEmbeddingCacheMetaPath()
        arrayPath: Path = getEmbeddingCacheArrayPath()

        try:
            try:
                import numpy as np #type: ignore
            except Exception:
                return

            np.save(str(arrayPath), embeddings, allow_pickle=False)

            meta: Dict[str, Any] = {
                "cacheKey": cacheKey,
                "modelName": self.modelName,
                "rows": int(len(self.giftRows)),
            }
            with metaPath.open("w", encoding="utf-8") as metaFile:
                json.dump(meta, metaFile, indent=4)
        except Exception:
            pass

    def setGiftCatalog(self, gifts: List[Dict[str, str]]) -> None:
        self.giftRows = list(gifts)
        self.giftTexts = [self._giftText(giftRow) for giftRow in self.giftRows]

        if not self.giftRows:
            self.giftEmbeddings = None
            return

        cacheKey: str = self._buildCacheKey(self.giftTexts)
        cachedEmbeddings: Optional[Any] = self._tryLoadEmbeddingsFromCache(cacheKey)
        if cachedEmbeddings is not None:
            self.giftEmbeddings = cachedEmbeddings
            return

        self.giftEmbeddings = None

    def ensureEmbeddingsReady(self) -> None:
        if not self.giftRows:
            self.giftEmbeddings = None
            return
        if self.giftEmbeddings is not None:
            return

        self.ensureReady()

        cacheKey: str = self._buildCacheKey(self.giftTexts)
        cachedEmbeddings: Optional[Any] = self._tryLoadEmbeddingsFromCache(cacheKey)
        if cachedEmbeddings is not None:
            self.giftEmbeddings = cachedEmbeddings
            return

        self.giftEmbeddings = self.model.encode(self.giftTexts, normalize_embeddings=True)
        self._saveEmbeddingsToCache(cacheKey, self.giftEmbeddings)

    def scoreToHappinessLikelihood(self, matchScore: float, lastYearRating: int) -> float:
        self.ensureReady()
        np = self.np

        center: float = 0.30
        steepness: float = 12.0
        baseProb: float = float(1.0 / (1.0 + np.exp(-(matchScore - center) * steepness)))

        ratingAdj: float = ((max(0, min(5, lastYearRating)) - 2.5) / 2.5) * 0.08
        prob: float = baseProb + ratingAdj
        prob = max(0.0, min(1.0, prob))
        return prob * 100.0

    def suggestBestGift(self, *, childProfile: Dict[str, str], rejectedGifts: Optional[set[str]] = None) -> Optional[Dict[str, str]]:
        self.ensureEmbeddingsReady()
        if not self.giftRows or self.giftEmbeddings is None:
            return None

        rejected: set[str] = rejectedGifts or set()

        primaryInterest: str = childProfile.get("primaryInterest", "")
        secondaryInterest: str = childProfile.get("secondaryInterest", "")
        wishlist: str = childProfile.get("wishlist", "")
        lastYearGift: str = childProfile.get("lastYearGift", "")
        ratingStr: str = childProfile.get("giftSatisfactionRating", "")

        try:
            lastYearRating: int = int(float(ratingStr)) if ratingStr.strip() != "" else 0
        except Exception:
            lastYearRating = 0

        interestsText: str = (primaryInterest + " " + secondaryInterest).strip()
        queryParts: List[str] = []

        if interestsText:
            queryParts.extend([interestsText] * 4)
        if wishlist:
            queryParts.extend([wishlist] * 3)

        if lastYearGift:
            if lastYearRating >= 4:
                queryParts.extend([f"liked last year: {lastYearGift}"] * 2)
            elif lastYearRating <= 2:
                queryParts.append(f"disliked last year: {lastYearGift}")

        baseQuery: str = " ".join(queryParts).strip()
        if not baseQuery:
            baseQuery = interestsText or wishlist or "gift ideas"

        self.ensureReady()
        childVec: Any = self.model.encode([baseQuery], normalize_embeddings=True)[0]
        rawScores: Any = self.giftEmbeddings @ childVec

        penaltyVec: Optional[Any] = None
        penaltyWeight: float = 0.0
        if lastYearGift and lastYearRating <= 2:
            penaltyVec = self.model.encode([f"similar to: {lastYearGift}"], normalize_embeddings=True)[0]
            penaltyWeight = 0.35

        bestGiftIndex: Optional[int] = None
        bestScore: float = -1e9

        for giftIndex, giftRow in enumerate(self.giftRows):
            giftName: str = giftRow.get("gift", "")
            if not giftName:
                continue
            if giftName in rejected:
                continue

            score: float = float(rawScores[giftIndex])

            if penaltyVec is not None and penaltyWeight > 0.0:
                simToBad: float = float(self.giftEmbeddings[giftIndex] @ penaltyVec)
                score -= penaltyWeight * simToBad

            if score > bestScore:
                bestScore = score
                bestGiftIndex = giftIndex

        if bestGiftIndex is None:
            return None

        chosenGift: Dict[str, str] = dict(self.giftRows[bestGiftIndex])
        chosenGift["matchScore"] = f"{bestScore:.4f}"
        chosenGift["happinessLikelihood"] = f"{self.scoreToHappinessLikelihood(bestScore, lastYearRating):.0f}"
        return chosenGift

    def scoreAllGiftsForChild(self, childProfile: Dict[str, str]) -> List[Tuple[Dict[str, str], float]]:
        self.ensureEmbeddingsReady()
        if not self.giftRows or self.giftEmbeddings is None:
            return []

        primaryInterest: str = childProfile.get("primaryInterest", "")
        secondaryInterest: str = childProfile.get("secondaryInterest", "")
        wishlist: str = childProfile.get("wishlist", "")
        lastYearGift: str = childProfile.get("lastYearGift", "")
        ratingStr: str = childProfile.get("giftSatisfactionRating", "")

        try:
            lastYearRating: int = int(float(ratingStr)) if ratingStr.strip() != "" else 0
        except Exception:
            lastYearRating = 0

        interestsText: str = (primaryInterest + " " + secondaryInterest).strip()
        queryParts: List[str] = []
        if interestsText:
            queryParts.extend([interestsText] * 4)
        if wishlist:
            queryParts.extend([wishlist] * 3)
        if lastYearGift:
            if lastYearRating >= 4:
                queryParts.extend([f"liked last year: {lastYearGift}"] * 2)
            elif lastYearRating <= 2:
                queryParts.append(f"disliked last year: {lastYearGift}")

        baseQuery: str = " ".join(queryParts).strip()
        if not baseQuery:
            baseQuery = interestsText or wishlist or "gift ideas"

        self.ensureReady()
        childVec: Any = self.model.encode([baseQuery], normalize_embeddings=True)[0]
        rawScores: Any = self.giftEmbeddings @ childVec

        penaltyVec: Optional[Any] = None
        penaltyWeight: float = 0.0
        if lastYearGift and lastYearRating <= 2:
            penaltyVec = self.model.encode([f"similar to: {lastYearGift}"], normalize_embeddings=True)[0]
            penaltyWeight = 0.35

        scored: List[Tuple[Dict[str, str], float]] = []
        for giftIndex, giftRow in enumerate(self.giftRows):
            giftName: str = giftRow.get("gift", "")
            if not giftName:
                continue

            score: float = float(rawScores[giftIndex])
            if penaltyVec is not None and penaltyWeight > 0.0:
                simToBad: float = float(self.giftEmbeddings[giftIndex] @ penaltyVec)
                score -= penaltyWeight * simToBad

            scored.append((dict(giftRow), score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class app(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.settings: appSettings = loadAppSettingsFromDisk()

        self.uiThemeChoices: Dict[str, str] = {}
        self.plotThemeChoices: Dict[str, str] = {}

        ctk.set_appearance_mode(self.settings.appearanceMode)

        self.buildUiThemeChoices()
        self.buildPlotThemeChoices()

        self.applyUiTheme(self.uiThemeChoices.get(self.settings.uiTheme, self.settings.uiTheme))

        self.title("Santa's Intelligent Gift Recommendation System")
        self.geometry("1150x740")

        self.tabView: ctk.CTkTabview = ctk.CTkTabview(self)
        self.tabView.pack(expand=True, fill="both", padx=10, pady=10)
        self.tabView.add("Main")
        self.tabView.add("Wishlist")
        self.tabView.add("Last Year")
        self.tabView.add("Child Interests")
        self.tabView.add("Settings")

        self.mainTabFrame: ctk.CTkFrame = self.tabView.tab("Main")
        self.wishlistTabFrame: ctk.CTkFrame = self.tabView.tab("Wishlist")
        self.lastYearTabFrame: ctk.CTkFrame = self.tabView.tab("Last Year")
        self.childInterestsTabFrame: ctk.CTkFrame = self.tabView.tab("Child Interests")
        self.settingsTabFrame: ctk.CTkFrame = self.tabView.tab("Settings")

        self.recommender: giftRecommender = giftRecommender()

        self.rejectedByChildId: Dict[str, set[str]] = {}
        self.finalChoicesByChildId: Dict[str, str] = {}

        self.childDisplayToId: Dict[str, str] = {}
        self.childDisplayValues: List[str] = []

        self.loadedChildId: Optional[str] = None
        self.loadedChildProfile: Dict[str, str] = {}
        self.loadedSuggestion: Optional[Dict[str, str]] = None

        self.buildMainTab()
        self.buildWishlistTab()
        self.buildLastYearTab()
        self.buildChildInterestsTab()
        self.buildSettingsTab()

        self.bind("<MouseWheel>", self.onMouseWheel, add="+")
        self.bind("<Button-4>", self.onMouseWheel, add="+")
        self.bind("<Button-5>", self.onMouseWheel, add="+")
        self.bind("<Prior>", self.onPageUp, add="+")
        self.bind("<Next>", self.onPageDown, add="+")
        self.bind("<Home>", self.onHome, add="+")
        self.bind("<End>", self.onEnd, add="+")

        self.finalChoicesByChildId = self.loadFinalListCsv()
        self.loadGiftCatalogFromDb()

        self.loadWishlistFromDb()
        self.loadLastYearFromDb()
        self.loadChildInterestsFromDb()
        self.refreshChildDropdown()

        self.applyPlotTheme(self.plotThemeChoices.get(self.settings.plotTheme, ""))

    def getDbPath(self) -> Path:
        return Path("./DB/childrensData.db")

    def currentTable(self) -> Optional[virtualizedTable]:
        currentTab: str = self.tabView.get()
        if currentTab == "Wishlist":
            return self.wishlistTable
        if currentTab == "Last Year":
            return self.lastYearTable
        if currentTab == "Child Interests":
            return self.childInterestsTable
        return None

    def onMouseWheel(self, event: Any) -> None:
        table: Optional[virtualizedTable] = self.currentTable()
        if table is not None:
            table.onMouseWheel(event)

    def onPageUp(self, _e: Any) -> None:
        table: Optional[virtualizedTable] = self.currentTable()
        if table is not None:
            table.scrollPages(-1)

    def onPageDown(self, _e: Any) -> None:
        table: Optional[virtualizedTable] = self.currentTable()
        if table is not None:
            table.scrollPages(1)

    def onHome(self, _e: Any) -> None:
        table: Optional[virtualizedTable] = self.currentTable()
        if table is not None:
            table.scrollHome()

    def onEnd(self, _e: Any) -> None:
        table: Optional[virtualizedTable] = self.currentTable()
        if table is not None:
            table.scrollEnd()

    def buildMainTab(self) -> None:
        outer: ctk.CTkFrame = ctk.CTkFrame(self.mainTabFrame)
        outer.pack(fill="both", expand=True, padx=12, pady=12)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)

        topRow: ctk.CTkFrame = ctk.CTkFrame(outer)
        topRow.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        topRow.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(topRow, text="Select Child:", anchor="w").grid(row=0, column=0, padx=(10, 8), pady=10)

        self.childSelectVar: ctk.StringVar = ctk.StringVar(value="(loading...)")
        self.childSelectMenu: ctk.CTkOptionMenu = ctk.CTkOptionMenu(topRow, variable=self.childSelectVar, values=["(loading...)"], command=self.onChildSelected, width=560)
        self.childSelectMenu.grid(row=0, column=1, padx=(0, 8), pady=10, sticky="w")

        self.loadChildButton: ctk.CTkButton = ctk.CTkButton(topRow, text="Load", command=self.loadSelectedChild, width=90)
        self.loadChildButton.grid(row=0, column=2, padx=(0, 8), pady=10)

        self.nextChildButton: ctk.CTkButton = ctk.CTkButton(topRow, text="Next", command=self.loadNextChild, width=90)
        self.nextChildButton.grid(row=0, column=3, padx=(0, 10), pady=10)

        statsFrame: ctk.CTkFrame = ctk.CTkFrame(outer)
        statsFrame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        statsFrame.grid_columnconfigure(1, weight=1)
        statsFrame.grid_columnconfigure(3, weight=1)

        self.mainIdVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainNameVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainWishlistVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainSubmittedVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainLastYearGiftVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainSatisfactionVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainPrimaryInterestVar: ctk.StringVar = ctk.StringVar(value="-")
        self.mainSecondaryInterestVar: ctk.StringVar = ctk.StringVar(value="-")

        ctk.CTkLabel(statsFrame, text="ID:", anchor="e").grid(row=0, column=0, padx=(10, 6), pady=(10, 6), sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainIdVar, anchor="w").grid(row=0, column=1, padx=(0, 10), pady=(10, 6), sticky="w")

        ctk.CTkLabel(statsFrame, text="Name:", anchor="e").grid(row=0, column=2, padx=(10, 6), pady=(10, 6), sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainNameVar, anchor="w").grid(row=0, column=3, padx=(0, 10), pady=(10, 6), sticky="w")

        ctk.CTkLabel(statsFrame, text="Submitted:", anchor="e").grid(row=1, column=0, padx=(10, 6), pady=6, sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainSubmittedVar, anchor="w").grid(row=1, column=1, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(statsFrame, text="Wishlist:", anchor="e").grid(row=2, column=0, padx=(10, 6), pady=6, sticky="ne")
        ctk.CTkLabel(statsFrame, textvariable=self.mainWishlistVar, anchor="w", justify="left").grid(row=2, column=1, columnspan=3, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(statsFrame, text="Last Year Gift:", anchor="e").grid(row=3, column=0, padx=(10, 6), pady=6, sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainLastYearGiftVar, anchor="w").grid(row=3, column=1, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(statsFrame, text="Satisfaction:", anchor="e").grid(row=3, column=2, padx=(10, 6), pady=6, sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainSatisfactionVar, anchor="w").grid(row=3, column=3, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(statsFrame, text="Primary Interest:", anchor="e").grid(row=4, column=0, padx=(10, 6), pady=6, sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainPrimaryInterestVar, anchor="w").grid(row=4, column=1, padx=(0, 10), pady=6, sticky="w")

        ctk.CTkLabel(statsFrame, text="Secondary Interest:", anchor="e").grid(row=4, column=2, padx=(10, 6), pady=6, sticky="e")
        ctk.CTkLabel(statsFrame, textvariable=self.mainSecondaryInterestVar, anchor="w").grid(row=4, column=3, padx=(0, 10), pady=6, sticky="w")

        actionFrame: ctk.CTkFrame = ctk.CTkFrame(outer)
        actionFrame.grid(row=2, column=0, sticky="nsew")
        actionFrame.grid_columnconfigure(0, weight=1)
        actionFrame.grid_rowconfigure(2, weight=1)

        buttonRow: ctk.CTkFrame = ctk.CTkFrame(actionFrame)
        buttonRow.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.generateSuggestionButton: ctk.CTkButton = ctk.CTkButton(buttonRow, text="Generate Suggestion", command=self.generateSuggestionForLoadedChild, width=190)
        self.generateSuggestionButton.pack(side="left", padx=(0, 8))

        self.acceptSuggestionButton: ctk.CTkButton = ctk.CTkButton(buttonRow, text="Accept (Save to CSV)", command=self.acceptSuggestion, width=190, state="disabled")
        self.acceptSuggestionButton.pack(side="left", padx=(0, 8))

        self.rejectSuggestionButton: ctk.CTkButton = ctk.CTkButton(buttonRow, text="Reject (Try Another)", command=self.rejectSuggestion, width=170, state="disabled")
        self.rejectSuggestionButton.pack(side="left", padx=(0, 8))

        self.explainGraphButton: ctk.CTkButton = ctk.CTkButton(buttonRow, text="Graph", command=self.openExplainGraphWindow, width=150)
        self.explainGraphButton.pack(side="left", padx=(0, 8))

        self.generateAllButton: ctk.CTkButton = ctk.CTkButton(buttonRow, text="Generate All", command=self.generateAllSuggestions, width=140)
        self.generateAllButton.pack(side="right")

        self.suggestionVar: ctk.StringVar = ctk.StringVar(value="Suggestion: -")
        self.suggestionReasonVar: ctk.StringVar = ctk.StringVar(value="")

        ctk.CTkLabel(actionFrame, textvariable=self.suggestionVar, anchor="w").grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkLabel(actionFrame, textvariable=self.suggestionReasonVar, anchor="w", justify="left").grid(row=2, column=0, sticky="new", padx=10, pady=(0, 10))

    def buildWishlistTab(self) -> None:
        self.wishlistTable = virtualizedTable(
            self.wishlistTabFrame,
            columns=[
                ("id", "ID", 25),
                ("name", "Name", 40),
                ("wishlist", "Wishlist", 80),
                ("submitted", "Submitted", 50),
            ],
        )
        self.wishlistTable.pack(fill="both", expand=True, padx=12, pady=12)

    def buildLastYearTab(self) -> None:
        self.lastYearTable = virtualizedTable(
            self.lastYearTabFrame,
            columns=[
                ("id", "ID", 25),
                ("name", "Name", 40),
                ("lastYearGift", "Last Year Gift", 80),
                ("giftSatisfactionRating", "Satisfaction", 40),
            ],
        )
        self.lastYearTable.pack(fill="both", expand=True, padx=12, pady=12)

    def buildChildInterestsTab(self) -> None:
        self.childInterestsTable = virtualizedTable(
            self.childInterestsTabFrame,
            columns=[
                ("id", "ID", 25),
                ("name", "Name", 40),
                ("primaryInterest", "Primary Interest", 80),
                ("secondaryInterest", "Secondary Interest", 80),
            ],
        )
        self.childInterestsTable.pack(fill="both", expand=True, padx=12, pady=12)

    def buildSettingsTab(self) -> None:
        settingsContainer: ctk.CTkFrame = ctk.CTkFrame(master=self.settingsTabFrame)
        settingsContainer.pack(pady=20, padx=20, fill="both", expand=True)
        settingsContainer.grid_columnconfigure(1, weight=1)

        appearanceLabel: ctk.CTkLabel = ctk.CTkLabel(master=settingsContainer, text="Appearance mode:", font=("Arial", 14))
        appearanceLabel.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="w")

        self.appearanceModeSegmented: ctk.CTkSegmentedButton = ctk.CTkSegmentedButton(master=settingsContainer, values=["Light", "Dark"], command=self.onAppearanceModeChanged)
        self.appearanceModeSegmented.set("Light" if self.settings.appearanceMode.lower() == "light" else "Dark")
        self.appearanceModeSegmented.grid(row=0, column=1, padx=10, pady=(10, 6), sticky="w")

        uiThemeLabel: ctk.CTkLabel = ctk.CTkLabel(master=settingsContainer, text="UI theme:", font=("Arial", 14))
        uiThemeLabel.grid(row=1, column=0, padx=10, pady=(12, 6), sticky="w")

        uiThemeValues: List[str] = self.buildUiThemeChoices()
        self.uiThemeVar: ctk.StringVar = ctk.StringVar(value=self.settings.uiTheme)
        self.uiThemeMenu: ctk.CTkOptionMenu = ctk.CTkOptionMenu(master=settingsContainer, values=uiThemeValues, variable=self.uiThemeVar, command=self.onUiThemeChanged, width=420)
        if self.settings.uiTheme not in uiThemeValues and uiThemeValues:
            self.uiThemeVar.set(uiThemeValues[0])
        self.uiThemeMenu.grid(row=1, column=1, padx=10, pady=(12, 6), sticky="w")

        plotThemeLabel: ctk.CTkLabel = ctk.CTkLabel(master=settingsContainer, text="Matplotlib theme:", font=("Arial", 14))
        plotThemeLabel.grid(row=2, column=0, padx=10, pady=(12, 6), sticky="w")

        plotThemeValues: List[str] = self.buildPlotThemeChoices()
        self.plotThemeVar: ctk.StringVar = ctk.StringVar(value=self.settings.plotTheme or "(default)")
        self.plotThemeMenu: ctk.CTkOptionMenu = ctk.CTkOptionMenu(master=settingsContainer, values=plotThemeValues, variable=self.plotThemeVar, command=self.onPlotThemeChanged, width=420)
        if self.plotThemeVar.get() not in plotThemeValues and plotThemeValues:
            self.plotThemeVar.set(plotThemeValues[0])
        self.plotThemeMenu.grid(row=2, column=1, padx=10, pady=(12, 6), sticky="w")

        hintLabel: ctk.CTkLabel = ctk.CTkLabel(master=settingsContainer, text=f"Custom themes folder:\n{getThemeDirPath()}", justify="left", anchor="w")
        hintLabel.grid(row=3, column=0, columnspan=2, padx=10, pady=(18, 10), sticky="w")

    def buildUiThemeChoices(self) -> List[str]:
        builtInThemes: List[str] = ["blue", "green", "dark-blue"]
        choices: Dict[str, str] = {themeName: themeName for themeName in builtInThemes}

        uiDir: Path = getUiThemeDirPath()
        for themePath in sorted(uiDir.glob("*.json")):
            displayName: str = f"custom:{themePath.stem}"
            choices[displayName] = str(themePath)

        self.uiThemeChoices = choices
        return list(choices.keys())

    def buildPlotThemeChoices(self) -> List[str]:
        choices: Dict[str, str] = {"(default)": ""}

        plotDir: Path = getPlotThemeDirPath()
        for themePath in sorted(plotDir.glob("*.json")):
            displayName: str = f"Aquarel:{themePath.stem}"
            choices[displayName] = str(themePath)

        commonMatplotlib: List[str] = ["classic", "ggplot"]
        for styleName in commonMatplotlib:
            choices[f"Matplotlib:{styleName}"] = styleName

        self.plotThemeChoices = choices
        return list(choices.keys())

    def onAppearanceModeChanged(self, value: str) -> None:
        normalizedValue: str = value.lower()
        self.settings.appearanceMode = "light" if normalizedValue == "light" else "dark"
        ctk.set_appearance_mode(self.settings.appearanceMode)
        saveAppSettingsToDisk(self.settings)

    def onUiThemeChanged(self, displayName: str) -> None:
        resolved: str = self.uiThemeChoices.get(displayName, displayName)
        self.settings.uiTheme = displayName
        self.applyUiTheme(resolved)
        saveAppSettingsToDisk(self.settings)

    def onPlotThemeChanged(self, displayName: str) -> None:
        resolved: str = self.plotThemeChoices.get(displayName, "")
        self.settings.plotTheme = displayName if displayName != "(default)" else ""
        self.applyPlotTheme(resolved)
        saveAppSettingsToDisk(self.settings)

    def applyUiTheme(self, themeValue: str) -> None:
        try:
            if themeValue and themeValue.lower().endswith(".json"):
                ctk.set_default_color_theme(themeValue)
                return
            if themeValue:
                ctk.set_default_color_theme(themeValue)
                return
        except Exception:
            pass

        try:
            ctk.set_default_color_theme("blue")
        except Exception:
            pass

    def applyPlotTheme(self, themeValue: str) -> None:
        try:
            import matplotlib as mpl #type: ignore
            import matplotlib.pyplot as plt
        except Exception:
            return

        mpl.rcParams.update(mpl.rcParamsDefault)

        if not themeValue:
            return

        if themeValue.lower().endswith(".json"):
            try:
                from aquarel import Theme #type: ignore
                theme = Theme.from_file(themeValue)
                theme.apply()
                return
            except Exception:
                mpl.rcParams.update(mpl.rcParamsDefault)
                return

        try:
            plt.style.use(themeValue)
        except Exception:
            mpl.rcParams.update(mpl.rcParamsDefault)

    def loadFinalListCsv(self) -> Dict[str, str]:
        path: Path = getFinalListCsvPath()
        result: Dict[str, str] = {}
        if not path.exists():
            return result

        try:
            with path.open("r", encoding="utf-8", newline="") as fileObj:
                reader = csv.DictReader(fileObj)
                for row in reader:
                    childId: str = str(row.get("id", "")).strip()
                    bestGift: str = str(row.get("best_gift", "")).strip()
                    if childId:
                        result[childId] = bestGift
        except Exception:
            traceback.print_exc()

        return result

    def writeFinalListCsv(self, rowsByChildId: Dict[str, Dict[str, str]]) -> None:
        path: Path = getFinalListCsvPath()
        tmpPath: Path = path.with_suffix(".tmp")

        def sortKey(childIdStr: str) -> int:
            try:
                return int(childIdStr)
            except Exception:
                return 10**9

        try:
            with tmpPath.open("w", encoding="utf-8", newline="") as fileObj:
                writer = csv.DictWriter(fileObj, fieldnames=["id", "name", "best_gift"])
                writer.writeheader()
                for childId in sorted(rowsByChildId.keys(), key=sortKey):
                    row = rowsByChildId[childId]
                    writer.writerow({"id": row.get("id", ""), "name": row.get("name", ""), "best_gift": row.get("bestGift", "")})
            tmpPath.replace(path)
        except Exception:
            traceback.print_exc()
            try:
                if tmpPath.exists():
                    tmpPath.unlink()
            except Exception:
                pass

    def refreshChildDropdown(self) -> None:
        children: List[Dict[str, str]] = self.loadDistinctChildrenFromDb()
        displayValues: List[str] = []
        displayToId: Dict[str, str] = {}

        for childRow in children:
            childId: str = childRow.get("id", "")
            childName: str = childRow.get("name", "")
            done: bool = childId in self.finalChoicesByChildId
            status: str = "Done" if done else "Todo"
            display: str = f"{childId} - {childName} [{status}]"
            displayValues.append(display)
            displayToId[display] = childId

        if not displayValues:
            displayValues = ["(no children found)"]
            displayToId = {"(no children found)": ""}

        self.childDisplayValues = displayValues
        self.childDisplayToId = displayToId

        self.childSelectMenu.configure(values=displayValues)
        if self.childSelectVar.get() not in displayValues:
            self.childSelectVar.set(displayValues[0])

    def onChildSelected(self, _value: str) -> None:
        self.loadedSuggestion = None
        self.acceptSuggestionButton.configure(state="disabled")
        self.rejectSuggestionButton.configure(state="disabled")
        self.suggestionVar.set("Suggestion: -")
        self.suggestionReasonVar.set("")

    def getSelectedChildIdFromMenu(self) -> str:
        display: str = self.childSelectVar.get()
        return self.childDisplayToId.get(display, "")

    def loadSelectedChild(self) -> None:
        childId: str = self.getSelectedChildIdFromMenu()
        if childId:
            self.loadChildById(childId)

    def loadNextChild(self) -> None:
        if not self.childDisplayValues:
            return

        currentDisplay: str = self.childSelectVar.get()
        try:
            currentIndex: int = self.childDisplayValues.index(currentDisplay)
        except ValueError:
            currentIndex = -1

        nextIndex: int = 0 if currentIndex < 0 else (currentIndex + 1) % len(self.childDisplayValues)
        self.childSelectVar.set(self.childDisplayValues[nextIndex])

        childId: str = self.getSelectedChildIdFromMenu()
        if childId:
            self.loadChildById(childId)

    def loadChildById(self, childId: str) -> None:
        profile: Dict[str, str] = self.loadChildProfileFromDb(childId)
        self.loadedChildId = childId
        self.loadedChildProfile = profile
        self.loadedSuggestion = None

        self.mainIdVar.set(profile.get("id", "-") or "-")
        self.mainNameVar.set(profile.get("name", "-") or "-")
        self.mainWishlistVar.set(profile.get("wishlist", "-") or "-")
        self.mainSubmittedVar.set(profile.get("submitted", "-") or "-")
        self.mainLastYearGiftVar.set(profile.get("lastYearGift", "-") or "-")
        self.mainSatisfactionVar.set(profile.get("giftSatisfactionRating", "-") or "-")
        self.mainPrimaryInterestVar.set(profile.get("primaryInterest", "-") or "-")
        self.mainSecondaryInterestVar.set(profile.get("secondaryInterest", "-") or "-")

        self.suggestionVar.set("Suggestion: -")
        self.suggestionReasonVar.set("")
        self.acceptSuggestionButton.configure(state="disabled")
        self.rejectSuggestionButton.configure(state="disabled")

    def generateSuggestionForLoadedChild(self) -> None:
        if not self.loadedChildId:
            self.suggestionVar.set("Suggestion: (load a child first)")
            self.suggestionReasonVar.set("")
            return

        rejected: set[str] = self.rejectedByChildId.get(self.loadedChildId, set())

        try:
            suggestion: Optional[Dict[str, str]] = self.recommender.suggestBestGift(
                childProfile=self.loadedChildProfile,
                rejectedGifts=rejected,
            )
        except Exception as exc:
            self.suggestionVar.set("Suggestion: (error)")
            self.suggestionReasonVar.set(str(exc))
            return

        if suggestion is None:
            self.suggestionVar.set("Suggestion: (no gift found)")
            self.suggestionReasonVar.set("Check gifts table / rejected list.")
            self.acceptSuggestionButton.configure(state="disabled")
            self.rejectSuggestionButton.configure(state="disabled")
            return

        self.loadedSuggestion = suggestion

        giftName: str = suggestion.get("gift", "")
        category: str = suggestion.get("category", "")
        ageLimit: str = suggestion.get("ageLimit", "")
        likelihood: str = suggestion.get("happinessLikelihood", "")
        matchScore: str = suggestion.get("matchScore", "")

        extra: str = ""
        if likelihood:
            extra += f" | Happiness: {likelihood}%"
        if matchScore:
            extra += f" | Match: {matchScore}"

        self.suggestionVar.set(f"Suggestion: {giftName}")
        self.suggestionReasonVar.set(f"Category: {category} | Age limit: {ageLimit}{extra}")
        self.acceptSuggestionButton.configure(state="normal")
        self.rejectSuggestionButton.configure(state="normal")

    def acceptSuggestion(self) -> None:
        if not self.loadedChildId or not self.loadedSuggestion:
            return

        giftName: str = self.loadedSuggestion.get("gift", "")
        if not giftName:
            return

        self.finalChoicesByChildId[self.loadedChildId] = giftName

        rowsByChildId: Dict[str, Dict[str, str]] = {}
        for childId, bestGift in self.finalChoicesByChildId.items():
            profile: Dict[str, str] = self.loadChildProfileFromDb(childId)
            rowsByChildId[childId] = {"id": childId, "name": profile.get("name", ""), "bestGift": bestGift}

        self.writeFinalListCsv(rowsByChildId)
        self.refreshChildDropdown()

        self.acceptSuggestionButton.configure(state="disabled")
        self.rejectSuggestionButton.configure(state="disabled")
        self.suggestionReasonVar.set("Saved to final_list.csv")

    def rejectSuggestion(self) -> None:
        if not self.loadedChildId or not self.loadedSuggestion:
            return

        giftName: str = self.loadedSuggestion.get("gift", "")
        if not giftName:
            return

        rejected: set[str] = self.rejectedByChildId.get(self.loadedChildId, set())
        rejected.add(giftName)
        self.rejectedByChildId[self.loadedChildId] = rejected

        self.loadedSuggestion = None
        self.acceptSuggestionButton.configure(state="disabled")
        self.rejectSuggestionButton.configure(state="disabled")
        self.suggestionVar.set("Suggestion: (rejected, generating new...)")
        self.suggestionReasonVar.set("")
        self.generateSuggestionForLoadedChild()

    def generateAllSuggestions(self) -> None:
        children: List[Dict[str, str]] = self.loadDistinctChildrenFromDb()
        if not children:
            self.suggestionVar.set("Suggestion: (no children found)")
            self.suggestionReasonVar.set("")
            return

        generatedCount: int = 0
        for childRow in children:
            childId: str = childRow.get("id", "")
            if not childId:
                continue
            if childId in self.finalChoicesByChildId:
                continue

            profile: Dict[str, str] = self.loadChildProfileFromDb(childId)
            rejected: set[str] = self.rejectedByChildId.get(childId, set())

            try:
                suggestion: Optional[Dict[str, str]] = self.recommender.suggestBestGift(childProfile=profile, rejectedGifts=rejected)
            except Exception:
                suggestion = None

            if suggestion is None:
                continue

            giftName: str = suggestion.get("gift", "")
            if not giftName:
                continue

            self.finalChoicesByChildId[childId] = giftName
            generatedCount += 1

        rowsByChildId: Dict[str, Dict[str, str]] = {}
        for childId, bestGift in self.finalChoicesByChildId.items():
            profile: Dict[str, str] = self.loadChildProfileFromDb(childId)
            rowsByChildId[childId] = {"id": childId, "name": profile.get("name", ""), "bestGift": bestGift}

        self.writeFinalListCsv(rowsByChildId)
        self.refreshChildDropdown()
        self.suggestionVar.set(f"Suggestion: Generated {generatedCount}")
        self.suggestionReasonVar.set("Saved to final_list.csv")

    def loadGiftCatalogFromDb(self) -> None:
        con: Optional[sqlite3.Connection] = None
        gifts: List[Dict[str, str]] = []

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT
                    gift,
                    age_limit,
                    category
                FROM gifts
                ORDER BY category COLLATE NOCASE ASC, gift COLLATE NOCASE ASC
                """
            )

            for row in cur.fetchall():
                gifts.append(
                    {
                        "gift": str(row["gift"] or ""),
                        "ageLimit": str(row["age_limit"] or ""),
                        "category": str(row["category"] or ""),
                    }
                )

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        try:
            # Uses cache if available, otherwise defers embedding build until first use.
            self.recommender.setGiftCatalog(gifts)
        except Exception as exc:
            self.suggestionVar.set("Suggestion: (embedding setup error)")
            self.suggestionReasonVar.set(str(exc))

    def loadDistinctChildrenFromDb(self) -> List[Dict[str, str]]:
        con: Optional[sqlite3.Connection] = None
        out: List[Dict[str, str]] = []

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT child_id, name
                FROM wishlist
                GROUP BY child_id, name
                ORDER BY CAST(child_id AS INTEGER) ASC
                """
            )

            for row in cur.fetchall():
                out.append({"id": str(row["child_id"] or ""), "name": str(row["name"] or "")})

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        return out

    def loadChildProfileFromDb(self, childId: str) -> Dict[str, str]:
        con: Optional[sqlite3.Connection] = None
        profile: Dict[str, str] = {
            "id": childId,
            "name": "",
            "wishlist": "",
            "submitted": "",
            "lastYearGift": "",
            "giftSatisfactionRating": "",
            "primaryInterest": "",
            "secondaryInterest": "",
        }

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT
                    child_id,
                    name,
                    GROUP_CONCAT(wishlist_items, ', ') AS wishlist,
                    MAX(submitted_date) AS submitted
                FROM wishlist
                WHERE CAST(child_id AS INTEGER) = CAST(? AS INTEGER)
                GROUP BY child_id, name
                """,
                (childId,),
            )
            row = cur.fetchone()
            if row is not None:
                profile["name"] = str(row["name"] or "")
                profile["wishlist"] = str(row["wishlist"] or "")
                profile["submitted"] = str(row["submitted"] or "")

            cur.execute(
                """
                SELECT
                    child_id,
                    last_year_gift,
                    gift_satisfaction_rating
                FROM history
                WHERE CAST(child_id AS INTEGER) = CAST(? AS INTEGER)
                """,
                (childId,),
            )
            row = cur.fetchone()
            if row is not None:
                profile["lastYearGift"] = str(row["last_year_gift"] or "")
                profile["giftSatisfactionRating"] = str(row["gift_satisfaction_rating"] or "")

            cur.execute(
                """
                SELECT
                    child_id,
                    primary_interest,
                    secondary_interest
                FROM intrests
                WHERE CAST(child_id AS INTEGER) = CAST(? AS INTEGER)
                """,
                (childId,),
            )
            row = cur.fetchone()
            if row is not None:
                profile["primaryInterest"] = str(row["primary_interest"] or "")
                profile["secondaryInterest"] = str(row["secondary_interest"] or "")

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        return profile

    def loadWishlistFromDb(self) -> None:
        items: List[Dict[str, str]] = []
        con: Optional[sqlite3.Connection] = None

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT
                    child_id,
                    name,
                    GROUP_CONCAT(wishlist_items, ', ') AS wishlist,
                    MAX(submitted_date) AS submitted
                FROM wishlist
                GROUP BY child_id, name
                ORDER BY CAST(child_id AS INTEGER) ASC
                """
            )

            for row in cur.fetchall():
                items.append(
                    {
                        "id": str(row["child_id"] or ""),
                        "name": str(row["name"] or ""),
                        "wishlist": str(row["wishlist"] or ""),
                        "submitted": str(row["submitted"] or ""),
                    }
                )

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        self.wishlistTable.setItems(items)

    def loadLastYearFromDb(self) -> None:
        items: List[Dict[str, str]] = []
        con: Optional[sqlite3.Connection] = None

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT
                    h.child_id AS id,
                    COALESCE(w.name, '') AS name,
                    COALESCE(h.last_year_gift, '') AS last_year_gift,
                    COALESCE(h.gift_satisfaction_rating, '') AS gift_satisfaction_rating
                FROM history h
                LEFT JOIN (
                    SELECT child_id AS id, name
                    FROM wishlist
                    GROUP BY child_id, name
                ) w
                ON CAST(w.id AS INTEGER) = CAST(h.child_id AS INTEGER)
                ORDER BY CAST(h.child_id AS INTEGER) ASC
                """
            )

            for row in cur.fetchall():
                items.append(
                    {
                        "id": str(row["id"] or ""),
                        "name": str(row["name"] or ""),
                        "lastYearGift": str(row["last_year_gift"] or ""),
                        "giftSatisfactionRating": str(row["gift_satisfaction_rating"] or ""),
                    }
                )

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        self.lastYearTable.setItems(items)

    def loadChildInterestsFromDb(self) -> None:
        items: List[Dict[str, str]] = []
        con: Optional[sqlite3.Connection] = None

        try:
            con = sqlite3.connect(str(self.getDbPath()))
            con.row_factory = sqlite3.Row
            cur: sqlite3.Cursor = con.cursor()

            cur.execute(
                """
                SELECT
                    i.child_id AS id,
                    COALESCE(w.name, '') AS name,
                    COALESCE(i.primary_interest, '') AS primary_interest,
                    COALESCE(i.secondary_interest, '') AS secondary_interest
                FROM intrests i
                LEFT JOIN (
                    SELECT child_id AS id, name
                    FROM wishlist
                    GROUP BY child_id, name
                ) w
                ON CAST(w.id AS INTEGER) = CAST(i.child_id AS INTEGER)
                ORDER BY CAST(i.child_id AS INTEGER) ASC
                """
            )

            for row in cur.fetchall():
                items.append(
                    {
                        "id": str(row["id"] or ""),
                        "name": str(row["name"] or ""),
                        "primaryInterest": str(row["primary_interest"] or ""),
                        "secondaryInterest": str(row["secondary_interest"] or ""),
                    }
                )

        except Exception:
            traceback.print_exc()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        self.childInterestsTable.setItems(items)

    def openExplainGraphWindow(self) -> None:
        if not self.loadedChildId:
            self.suggestionVar.set("Suggestion: (load a child first)")
            self.suggestionReasonVar.set("")
            return

        profile: Dict[str, str] = self.loadedChildProfile or self.loadChildProfileFromDb(self.loadedChildId)
        scored: List[Tuple[Dict[str, str], float]] = self.recommender.scoreAllGiftsForChild(profile)
        if not scored:
            self.suggestionVar.set("Suggestion: (no scores)")
            self.suggestionReasonVar.set("Check gifts table / embeddings.")
            return

        topCount: int = 10 if len(scored) >= 10 else len(scored)
        topScored: List[Tuple[Dict[str, str], float]] = scored[:topCount]

        window: ctk.CTkToplevel = ctk.CTkToplevel(self)
        window.title("Explain + Graph")
        window.geometry("980x620")
        window.grid_columnconfigure(0, weight=1)
        window.grid_columnconfigure(1, weight=1)
        window.grid_rowconfigure(0, weight=1)

        leftFrame: ctk.CTkFrame = ctk.CTkFrame(window)
        leftFrame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        leftFrame.grid_rowconfigure(1, weight=1)
        leftFrame.grid_columnconfigure(0, weight=1)

        rightFrame: ctk.CTkFrame = ctk.CTkFrame(window)
        rightFrame.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        rightFrame.grid_rowconfigure(0, weight=1)
        rightFrame.grid_columnconfigure(0, weight=1)

        titleVar: ctk.StringVar = ctk.StringVar(value=f"Child: {profile.get('id','')} - {profile.get('name','')}")
        ctk.CTkLabel(leftFrame, textvariable=titleVar, anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))

        explanationText: str = self.buildExplanationText(profile, topScored)
        textBox: ctk.CTkTextbox = ctk.CTkTextbox(leftFrame, wrap="word")
        textBox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        textBox.insert("1.0", explanationText)
        textBox.configure(state="disabled")

        self.renderScoreChart(rightFrame, profile, topScored)
        self.renderHappinessCurveChart(rightFrame, profile, topScored)

    def buildExplanationText(self, profile: Dict[str, str], topScored: List[Tuple[Dict[str, str], float]]) -> str:
        childId: str = profile.get("id", "")
        name: str = profile.get("name", "")
        wishlist: str = profile.get("wishlist", "")
        primaryInterest: str = profile.get("primaryInterest", "")
        secondaryInterest: str = profile.get("secondaryInterest", "")
        lastYearGift: str = profile.get("lastYearGift", "")
        ratingStr: str = profile.get("giftSatisfactionRating", "")

        try:
            rating: int = int(float(ratingStr)) if ratingStr.strip() != "" else 0
        except Exception:
            rating = 0

        rejected: set[str] = self.rejectedByChildId.get(childId, set())

        lines: List[str] = []
        lines.append("Profile")
        lines.append(f"- id: {childId}")
        lines.append(f"- name: {name}")
        lines.append(f"- wishlist text used: {wishlist or '(empty)'}")
        lines.append(f"- interests text used: {(primaryInterest + ' ' + secondaryInterest).strip() or '(empty)'}")
        lines.append(f"- last year gift: {lastYearGift or '(none)'}")
        lines.append(f"- last year satisfaction (0-5): {rating}")
        lines.append("")
        lines.append("Top matches (score is similarity after penalty): ")
        
        for rankIndex, (giftRow, scoreValue) in enumerate(topScored, start=1):
            giftName: str = giftRow.get("gift", "")
            category: str = giftRow.get("category", "")
            ageLimit: str = giftRow.get("ageLimit", "")
            happiness: float = self.recommender.scoreToHappinessLikelihood(scoreValue, rating)
            lines.append(f"{rankIndex}. {giftName} | score={scoreValue:.4f} | happiness≈{happiness:.0f}% | category={category} | ageLimit={ageLimit}")

        return "\n".join(lines)

    def renderScoreChart(self, masterFrame: ctk.CTkFrame, profile: Dict[str, str], topScored: List[Tuple[Dict[str, str], float]]) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            ctk.CTkLabel(masterFrame, text=f"Matplotlib not available:\n{exc}", justify="left",).grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            return

        themeKey: str = self.settings.plotTheme or "(default)"
        resolvedTheme: str = self.plotThemeChoices.get(themeKey, "")
        self.applyPlotTheme(resolvedTheme)

        giftNames: List[str] = []
        scores: List[float] = []
        for giftRow, scoreValue in topScored:
            giftNames.append(str(giftRow.get("gift", "")))
            scores.append(float(scoreValue))

        figure = plt.figure(figsize=(6.5, 5.2), dpi=100)
        ax = figure.add_subplot(111)

        ax.bar(giftNames, scores)

        ax.set_title("Top gift similarity scores (after penalty)")
        ax.set_xlabel("Gift")
        ax.set_ylabel("Similarity score")

        ax.tick_params(axis="x", labelrotation=35)
        for tickLabel in ax.get_xticklabels():
            tickLabel.set_horizontalalignment("right")
            tickLabel.set_rotation_mode("anchor")

        figure.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.32)

        canvas = FigureCanvasTkAgg(figure, master=masterFrame)
        canvasWidget = canvas.get_tk_widget()
        canvasWidget.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        canvas.draw()

    def renderHappinessCurveChart(self, masterFrame: ctk.CTkFrame, profile: Dict[str, str], topScored: List[Tuple[Dict[str, str], float]]) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            ctk.CTkLabel(
                masterFrame,
                text=f"Matplotlib not available:\n{exc}",
                justify="left",
            ).grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
            return

        themeKey: str = self.settings.plotTheme or "(default)"
        resolvedTheme: str = self.plotThemeChoices.get(themeKey, "")
        self.applyPlotTheme(resolvedTheme)

        ratingStr: str = profile.get("giftSatisfactionRating", "")
        try:
            rating: int = int(float(ratingStr)) if str(ratingStr).strip() != "" else 0
        except Exception:
            rating = 0

        xs: List[float] = []
        ys: List[float] = []

        steps: int = 18
        for stepIndex in range(steps):
            x: float = -0.05 + (0.90 * (float(stepIndex) / float(steps - 1)))
            y: float = float(self.recommender.scoreToHappinessLikelihood(x, rating))
            xs.append(x)
            ys.append(y)

        chosenScore: float = float(topScored[0][1]) if topScored else 0.0
        chosenHappy: float = float(self.recommender.scoreToHappinessLikelihood(chosenScore, rating))

        figure = plt.figure(figsize=(6.5, 2.8), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(xs, ys)
        ax.scatter([chosenScore], [chosenHappy], marker="X", s=120)

        ax.axvline(chosenScore, linestyle="--")
        ax.axhline(chosenHappy, linestyle="--")
        ax.annotate(
            f"chosen\n({chosenScore:.3f}, {chosenHappy:.0f}%)",
            xy=(chosenScore, chosenHappy),
            xytext=(8, 8),
            textcoords="offset points",
        )

        canvas = FigureCanvasTkAgg(figure, master=masterFrame)
        canvasWidget = canvas.get_tk_widget()
        canvasWidget.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        canvas.draw()

    def loadDistinctChildrenFromWishlistOnly(self) -> List[Dict[str, str]]:
        return self.loadDistinctChildrenFromDb()


if __name__ == "__main__":
    appInstance: app = app()
    appInstance.mainloop()
