import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, type CloudSttProvider } from "@/bindings";
import { Dropdown, SettingContainer, SettingsGroup } from "@/components/ui";
import { ApiKeyField } from "./PostProcessingSettingsApi/ApiKeyField";
import { BaseUrlField } from "./PostProcessingSettingsApi/BaseUrlField";
import { ModelSelect } from "./PostProcessingSettingsApi/ModelSelect";
import { useSettings } from "../../hooks/useSettings";

const LOCAL_PROVIDER_ID = "local";

/// Pick where speech is transcribed: the local model or a cloud API.
export const CloudTranscription: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, refreshSettings } = useSettings();
  const [providers, setProviders] = useState<CloudSttProvider[]>([]);

  useEffect(() => {
    commands.getCloudTranscriptionProviders().then((result) => {
      if (result.status === "ok") setProviders(result.data);
    });
  }, []);

  const providerId = getSetting("transcription_provider_id") ?? "local";
  const apiKeys = getSetting("transcription_api_keys") ?? {};
  const models = getSetting("transcription_models") ?? {};
  const baseUrls = getSetting("transcription_base_urls") ?? {};
  const provider = providers.find((p) => p.id === providerId);

  const providerOptions = useMemo(
    () => [
      {
        value: LOCAL_PROVIDER_ID,
        label: t("settings.cloudTranscription.local"),
      },
      ...providers.map((p) => ({ value: p.id, label: p.label })),
    ],
    [providers, t],
  );

  const modelOptions = useMemo(
    () => (provider?.models ?? []).map((m) => ({ value: m, label: m })),
    [provider],
  );

  const run = async (
    call: Promise<{ status: "ok" | "error"; error?: string }>,
  ) => {
    const result = await call;
    if (result.status === "error") console.error(result.error);
    await refreshSettings();
  };

  const model = provider ? models[provider.id] || provider.default_model : "";

  return (
    <SettingsGroup title={t("settings.cloudTranscription.title")}>
      <SettingContainer
        title={t("settings.cloudTranscription.provider.title")}
        description={t("settings.cloudTranscription.provider.description")}
        descriptionMode="tooltip"
        layout="horizontal"
        grouped={true}
      >
        <Dropdown
          options={providerOptions}
          selectedValue={providerId}
          onSelect={(value) => run(commands.setTranscriptionProvider(value))}
        />
      </SettingContainer>

      {provider && (
        <>
          {provider.allow_base_url_edit && (
            <SettingContainer
              title={t("settings.cloudTranscription.baseUrl.title")}
              description={t("settings.cloudTranscription.baseUrl.description")}
              descriptionMode="tooltip"
              layout="horizontal"
              grouped={true}
            >
              <BaseUrlField
                value={baseUrls[provider.id] ?? provider.base_url}
                onBlur={(value) =>
                  run(
                    commands.changeTranscriptionBaseUrlSetting(
                      provider.id,
                      value,
                    ),
                  )
                }
                placeholder={provider.base_url}
                disabled={false}
                className="min-w-[380px]"
              />
            </SettingContainer>
          )}

          <SettingContainer
            title={t("settings.cloudTranscription.apiKey.title")}
            description={t("settings.cloudTranscription.apiKey.description", {
              provider: provider.label,
            })}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <ApiKeyField
              value={apiKeys[provider.id] ?? ""}
              onBlur={(value) =>
                run(
                  commands.changeTranscriptionApiKeySetting(provider.id, value),
                )
              }
              placeholder={t("settings.cloudTranscription.apiKey.placeholder")}
              disabled={false}
              className="min-w-[320px]"
            />
          </SettingContainer>

          {provider.default_model && (
            <SettingContainer
              title={t("settings.cloudTranscription.model.title")}
              description={t("settings.cloudTranscription.model.description")}
              descriptionMode="tooltip"
              layout="stacked"
              grouped={true}
            >
              <ModelSelect
                value={model}
                options={modelOptions}
                disabled={false}
                placeholder={provider.default_model}
                onSelect={(value) =>
                  run(
                    commands.changeTranscriptionModelSetting(
                      provider.id,
                      value,
                    ),
                  )
                }
                onCreate={(value) =>
                  run(
                    commands.changeTranscriptionModelSetting(
                      provider.id,
                      value,
                    ),
                  )
                }
                onBlur={() => {}}
                className="flex-1 min-w-[380px]"
              />
            </SettingContainer>
          )}
        </>
      )}
    </SettingsGroup>
  );
};
