import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface VoiceCommandsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const VoiceCommands: React.FC<VoiceCommandsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const enabled = getSetting("voice_commands_enabled") ?? true;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("voice_commands_enabled", nextEnabled)
        }
        isUpdating={isUpdating("voice_commands_enabled")}
        label={t("settings.advanced.voiceCommands.title")}
        description={t("settings.advanced.voiceCommands.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
