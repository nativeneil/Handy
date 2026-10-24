import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface ContextAwarenessProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const ContextAwareness: React.FC<ContextAwarenessProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const enabled = getSetting("context_awareness_enabled") ?? true;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("context_awareness_enabled", nextEnabled)
        }
        isUpdating={isUpdating("context_awareness_enabled")}
        label={t("settings.postProcessing.context.awareness.title")}
        description={t("settings.postProcessing.context.awareness.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
