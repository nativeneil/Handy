import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface AlwaysPostProcessProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const AlwaysPostProcess: React.FC<AlwaysPostProcessProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const enabled = getSetting("always_post_process") ?? false;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(nextEnabled) =>
          updateSetting("always_post_process", nextEnabled)
        }
        isUpdating={isUpdating("always_post_process")}
        label={t("settings.postProcessing.context.always.title")}
        description={t("settings.postProcessing.context.always.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
