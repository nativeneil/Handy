import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TranscriptionProviderField } from "@/bindings";
import { Dropdown, SettingContainer, SettingsGroup } from "@/components/ui";
import { ApiKeyField } from "./PostProcessingSettingsApi/ApiKeyField";
import { BaseUrlField } from "./PostProcessingSettingsApi/BaseUrlField";
import { ModelSelect } from "./PostProcessingSettingsApi/ModelSelect";
import { LanguageSelector } from "./LanguageSelector";
import { useSettings } from "../../hooks/useSettings";
import {
  LOCAL_PROVIDER_ID,
  useCloudSttProviders,
} from "../../hooks/useCloudSttProviders";

/** Pick where speech is transcribed: the local model or a cloud API. */
export const CloudTranscription: React.FC = () => {
  const { t } = useTranslation();
  const {
    getSetting,
    updateSetting,
    updateTranscriptionProviderSetting,
    isUpdating,
  } = useSettings();
  const {
    providers,
    providerId,
    activeProvider: provider,
  } = useCloudSttProviders();

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

  if (!provider) {
    return (
      <SettingsGroup title={t("settings.cloudTranscription.title")}>
        <ProviderRow
          options={providerOptions}
          value={providerId}
          onSelect={(id) => updateSetting("transcription_provider_id", id)}
        />
      </SettingsGroup>
    );
  }

  const value = (field: TranscriptionProviderField) => {
    const key = {
      api_key: "transcription_api_keys",
      model: "transcription_models",
      base_url: "transcription_base_urls",
    } as const;
    return getSetting(key[field])?.[provider.id] ?? "";
  };
  const save = (field: TranscriptionProviderField) => (next: string) =>
    updateTranscriptionProviderSetting(field, provider.id, next);
  const busy = (field: TranscriptionProviderField) =>
    isUpdating(`transcription_${field}:${provider.id}`);

  return (
    <SettingsGroup title={t("settings.cloudTranscription.title")}>
      <ProviderRow
        options={providerOptions}
        value={providerId}
        onSelect={(id) => updateSetting("transcription_provider_id", id)}
      />

      {provider.allow_base_url_edit && (
        <SettingContainer
          title={t("settings.cloudTranscription.baseUrl.title")}
          description={t("settings.cloudTranscription.baseUrl.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <BaseUrlField
            value={value("base_url") || provider.base_url}
            onBlur={save("base_url")}
            placeholder={provider.base_url_placeholder}
            disabled={busy("base_url")}
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
          value={value("api_key")}
          onBlur={save("api_key")}
          placeholder={t(
            provider.requires_api_key
              ? "settings.cloudTranscription.apiKey.placeholder"
              : "settings.cloudTranscription.apiKey.placeholderOptional",
          )}
          disabled={busy("api_key")}
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
            value={value("model") || provider.default_model}
            options={modelOptions}
            disabled={busy("model")}
            placeholder={provider.default_model}
            onSelect={save("model")}
            onCreate={save("model")}
            onBlur={() => {}}
            className="flex-1 min-w-[380px]"
          />
        </SettingContainer>
      )}

      {/* Cloud APIs accept any language, so the local model's list doesn't apply. */}
      <LanguageSelector
        descriptionMode="tooltip"
        grouped={true}
        supportsLanguageDetection={true}
      />
    </SettingsGroup>
  );
};

const ProviderRow: React.FC<{
  options: { value: string; label: string }[];
  value: string;
  onSelect: (id: string) => void;
}> = ({ options, value, onSelect }) => {
  const { t } = useTranslation();
  return (
    <SettingContainer
      title={t("settings.cloudTranscription.provider.title")}
      description={t("settings.cloudTranscription.provider.description")}
      descriptionMode="tooltip"
      layout="horizontal"
      grouped={true}
    >
      <Dropdown options={options} selectedValue={value} onSelect={onSelect} />
    </SettingContainer>
  );
};
