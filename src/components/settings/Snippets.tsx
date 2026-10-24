import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import type { Snippet } from "@/bindings";
import { useSettings } from "../../hooks/useSettings";
import { Input } from "../ui/Input";
import { Textarea } from "../ui/Textarea";
import { Button } from "../ui/Button";
import { SettingContainer } from "../ui/SettingContainer";

interface SnippetsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

const MAX_TRIGGER_LENGTH = 60;
const MAX_EXPANSION_LENGTH = 4000;

const normalizeTrigger = (trigger: string) =>
  trigger.replace(/\s+/g, " ").trim();

export const Snippets: React.FC<SnippetsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const [trigger, setTrigger] = useState("");
    const [expansion, setExpansion] = useState("");
    const snippets: Snippet[] = getSetting("snippets") || [];
    const normalizedTrigger = normalizeTrigger(trigger);
    const updating = isUpdating("snippets");

    const canAdd =
      normalizedTrigger.length > 0 &&
      normalizedTrigger.length <= MAX_TRIGGER_LENGTH &&
      expansion.length > 0 &&
      expansion.length <= MAX_EXPANSION_LENGTH &&
      !updating;

    const handleAdd = () => {
      if (!canAdd) return;
      const duplicate = snippets.some(
        (s) => s.trigger.toLowerCase() === normalizedTrigger.toLowerCase(),
      );
      if (duplicate) {
        toast.error(
          t("settings.advanced.snippets.duplicate", {
            trigger: normalizedTrigger,
          }),
        );
        return;
      }
      updateSetting("snippets", [
        ...snippets,
        {
          id: `snippet_${Date.now()}`,
          trigger: normalizedTrigger,
          expansion,
        },
      ]);
      setTrigger("");
      setExpansion("");
    };

    const handleRemove = (id: string) => {
      updateSetting(
        "snippets",
        snippets.filter((s) => s.id !== id),
      );
    };

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.snippets.title")}
          description={t("settings.advanced.snippets.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
          layout="stacked"
        >
          <div className="flex flex-col gap-2">
            <Input
              type="text"
              value={trigger}
              onChange={(e) => setTrigger(e.target.value)}
              placeholder={t("settings.advanced.snippets.triggerPlaceholder")}
              variant="compact"
              maxLength={MAX_TRIGGER_LENGTH}
              disabled={updating}
            />
            <Textarea
              value={expansion}
              onChange={(e) => setExpansion(e.target.value)}
              placeholder={t("settings.advanced.snippets.expansionPlaceholder")}
              variant="compact"
              maxLength={MAX_EXPANSION_LENGTH}
              disabled={updating}
            />
            <div>
              <Button
                onClick={handleAdd}
                disabled={!canAdd}
                variant="primary"
                size="md"
              >
                {t("settings.advanced.snippets.add")}
              </Button>
            </div>
          </div>
        </SettingContainer>
        {snippets.length > 0 && (
          <div
            className={`px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"} flex flex-col gap-1`}
          >
            {snippets.map((snippet) => (
              <div
                key={snippet.id}
                className="flex items-start justify-between gap-3 py-1"
              >
                <div className="min-w-0">
                  <div className="text-sm font-semibold">{snippet.trigger}</div>
                  <div className="text-xs text-mid-gray whitespace-pre-wrap break-words line-clamp-3">
                    {snippet.expansion}
                  </div>
                </div>
                <Button
                  onClick={() => handleRemove(snippet.id)}
                  disabled={updating}
                  variant="secondary"
                  size="sm"
                  aria-label={t("settings.advanced.snippets.remove", {
                    trigger: snippet.trigger,
                  })}
                >
                  {t("settings.advanced.snippets.removeLabel")}
                </Button>
              </div>
            ))}
          </div>
        )}
      </>
    );
  },
);
