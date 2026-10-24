import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, type AppPromptRule, type RunningApp } from "@/bindings";
import { useSettings } from "../../hooks/useSettings";
import { Button } from "../ui/Button";
import { Dropdown } from "../ui/Dropdown";
import { SettingContainer } from "../ui/SettingContainer";

interface AppPromptsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

/** Per-app prompts: dictating into a listed app uses its prompt instead of
 * the selected one (macOS, matched by bundle identifier). */
export const AppPrompts: React.FC<AppPromptsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const rules: AppPromptRule[] = getSetting("app_prompt_rules") || [];
    const prompts = getSetting("post_process_prompts") || [];
    const updating = isUpdating("app_prompt_rules");

    const [runningApps, setRunningApps] = useState<RunningApp[]>([]);
    const [newApp, setNewApp] = useState<string | null>(null);
    const [newPrompt, setNewPrompt] = useState<string | null>(null);

    const loadRunningApps = useCallback(async () => {
      try {
        setRunningApps(await commands.getRunningApps());
      } catch {
        setRunningApps([]);
      }
    }, []);

    useEffect(() => {
      loadRunningApps();
    }, [loadRunningApps]);

    const promptOptions = prompts.map((p) => ({ value: p.id, label: p.name }));
    const appOptions = runningApps
      .filter((app) => !rules.some((r) => r.bundle_id === app.bundle_id))
      .map((app) => ({ value: app.bundle_id, label: app.name }));

    const setRules = (next: AppPromptRule[]) =>
      updateSetting("app_prompt_rules", next);

    const handleAdd = () => {
      const app = runningApps.find((a) => a.bundle_id === newApp);
      if (!app || !newPrompt) return;
      setRules([
        ...rules,
        { bundle_id: app.bundle_id, app_name: app.name, prompt_id: newPrompt },
      ]);
      setNewApp(null);
    };

    const handleChangePrompt = (bundleId: string, promptId: string) =>
      setRules(
        rules.map((r) =>
          r.bundle_id === bundleId ? { ...r, prompt_id: promptId } : r,
        ),
      );

    const handleRemove = (bundleId: string) =>
      setRules(rules.filter((r) => r.bundle_id !== bundleId));

    return (
      <>
        <SettingContainer
          title={t("settings.postProcessing.appPrompts.title")}
          description={t("settings.postProcessing.appPrompts.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
          layout="stacked"
        >
          <div className="flex flex-wrap items-center gap-2">
            <Dropdown
              options={appOptions}
              selectedValue={newApp}
              onSelect={setNewApp}
              onOpen={loadRunningApps}
              placeholder={t("settings.postProcessing.appPrompts.selectApp")}
              disabled={updating}
            />
            <Dropdown
              options={promptOptions}
              selectedValue={newPrompt}
              onSelect={setNewPrompt}
              placeholder={t("settings.postProcessing.appPrompts.selectPrompt")}
              disabled={updating}
            />
            <Button
              onClick={handleAdd}
              disabled={!newApp || !newPrompt || updating}
              variant="primary"
              size="md"
            >
              {t("settings.postProcessing.appPrompts.add")}
            </Button>
          </div>
        </SettingContainer>
        {rules.length > 0 && (
          <div
            className={`px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"} flex flex-col gap-1`}
          >
            {rules.map((rule) => (
              <div
                key={rule.bundle_id}
                className="flex items-center justify-between gap-3 py-1"
              >
                <span
                  className="text-sm font-semibold truncate"
                  title={rule.bundle_id}
                >
                  {rule.app_name}
                </span>
                <div className="flex items-center gap-2">
                  <Dropdown
                    options={promptOptions}
                    selectedValue={rule.prompt_id}
                    onSelect={(id) => handleChangePrompt(rule.bundle_id, id)}
                    disabled={updating}
                  />
                  <Button
                    onClick={() => handleRemove(rule.bundle_id)}
                    disabled={updating}
                    variant="secondary"
                    size="sm"
                    aria-label={t("settings.postProcessing.appPrompts.remove", {
                      app: rule.app_name,
                    })}
                  >
                    {t("settings.postProcessing.appPrompts.removeLabel")}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </>
    );
  },
);
