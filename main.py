from __future__ import annotations

import json
import random
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import customtkinter as ctk

@dataclass
class AppSettings:
    appearanceMode: str = "dark"
    colorTheme: str = "blue"

def getSettingsFilePath() -> Path:
    homeDirPath: Path = Path.home()
    docDirPath: Path = homeDirPath / "Documents"
    settingsDirPath: Path = docDirPath / "IGRS"
    settingsDirPath.mkdir(parents=True, exist_ok=True)
    settingsFilePath: Path = settingsDirPath / "settings.json"
    return settingsFilePath

def loadAppSettingsFromDisk() -> AppSettings:
    settingsPath: Path = getSettingsFilePath()

    if not settingsPath.exists():
        return AppSettings()
    
    try:
        with settingsPath.open("r", encoding="utf-8") as settingsFile:
            rawData = json.load(settingsFile)
        return AppSettings(appearanceMode=str(rawData.get("appearanceMode", "dark")), colorTheme=str(rawData.get("colorTheme", "blue")))
    
    except Exception as error:
        print("Error loading app settings: ", error)
        traceback.print_exc()
        return AppSettings()

def saveAppSettingsToDisk(appSettings: AppSettings) -> None:
    settingsFilePath: Path = getSettingsFilePath()

    try:
        with settingsFilePath.open("w", encoding="utf-8") as settingsFile:
            json.dump(asdict(appSettings), settingsFile, indent=4)
    except Exception as error:
        print(f"Error saving app settings: ", error)
        traceback.print_exc()


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.appSettings: AppSettings = loadAppSettingsFromDisk()

        ctk.set_appearance_mode(self.appSettings.appearanceMode)
        ctk.set_default_color_theme(self.appSettings.colorTheme)

        self.currentAppearanceMode: str = self.appSettings.appearanceMode
        self.currentColorTheme: str = self.appSettings.colorTheme

        self.title("Santa's Intelligent Gift Recommendation System")
        self.geometry(f"{500}x{500}")


        self.tabView: ctk.CTkTabview = ctk.CTkTabview(self)
        self.tabView.pack(expand=True, fill="both", padx=10, pady=10)
        self.tabView.add("Main")
        self.tabView.add("Settings")

        self.mainTabFrame: ctk.CTkFrame = self.tabView.tab("Main")
        self.settingsTabFrame: ctk.CTkFrame = self.tabView.tab("Settings")

        self.buildMainTab()
        self.buildSettingsTab()

    def buildMainTab(self) -> None:
        pass


    def buildSettingsTab(self) -> None:
        settingsContainer: ctk.CTkFrame = ctk.CTkFrame(master=self.settingsTabFrame)
        settingsContainer.pack(pady=20, padx=20, fill="both", expand=True)

        appearanceFrame: ctk.CTkFrame = ctk.CTkFrame(master=settingsContainer)
        appearanceFrame.pack(fill="x", pady=10)

        appearanceLabel: ctk.CTkLabel = ctk.CTkLabel(master=appearanceFrame, text="Appearance mode:", font=("Arial", 14))
        appearanceLabel.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.appearanceModeSegmented: ctk.CTkSegmentedButton = ctk.CTkSegmentedButton(master=appearanceFrame, values=["Light", "Dark"], command=self.onAppearanceModeChanged)
        if self.appSettings.appearanceMode.lower() == "light":
            self.appearanceModeSegmented.set("Light")
        else:
            self.appearanceModeSegmented.set("Dark")
        self.appearanceModeSegmented.grid(row=0, column=1, padx=10, pady=10)

        themeFrame: ctk.CTkFrame = ctk.CTkFrame(master=settingsContainer)
        themeFrame.pack(fill="x", pady=10)

        themeLabel: ctk.CTkLabel = ctk.CTkLabel(master=themeFrame, text="Color theme:", font=("Arial", 14))
        themeLabel.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.themeOptionMenu: ctk.CTkOptionMenu = ctk.CTkOptionMenu(master=themeFrame, values=["blue", "green", "dark-blue"], command=self.onThemeChanged)
        self.themeOptionMenu.set(self.appSettings.colorTheme)
        self.themeOptionMenu.grid(row=0, column=1, padx=10, pady=10)

    def saveSettings(self) -> None:
        saveAppSettingsToDisk(self.appSettings)

    def onAppearanceModeChanged(self, value: str) -> None:
        normalizedValue: str = value.lower()
        if normalizedValue == "light":
            self.currentAppearanceMode = "light"
        else:
            self.currentAppearanceMode = "dark"

        self.appSettings.appearanceMode = self.currentAppearanceMode
        ctk.set_appearance_mode(self.currentAppearanceMode)
        self.saveSettings()

    def onThemeChanged(self, value: str) -> None:
        self.currentColorTheme = value
        self.appSettings.colorTheme = value
        ctk.set_default_color_theme(self.currentColorTheme)
        self.saveSettings()


if __name__ == "__main__":
    app = App()
    app.mainloop()