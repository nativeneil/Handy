import { useEffect, useState } from "react";
import { commands, type CloudSttProvider } from "@/bindings";
import { useSettings } from "./useSettings";

/** Mirrors `cloud_stt::LOCAL_PROVIDER_ID`. */
export const LOCAL_PROVIDER_ID = "local";

// The catalog is static for the app's lifetime; fetch it once.
let catalog: Promise<CloudSttProvider[]> | null = null;

/** The cloud STT catalog and the provider the settings select, if any. */
export const useCloudSttProviders = () => {
  const { getSetting } = useSettings();
  const [providers, setProviders] = useState<CloudSttProvider[]>([]);

  useEffect(() => {
    catalog ??= commands.getCloudTranscriptionProviders();
    catalog.then(setProviders);
  }, []);

  const providerId =
    getSetting("transcription_provider_id") ?? LOCAL_PROVIDER_ID;
  const activeProvider = providers.find((p) => p.id === providerId) ?? null;
  return { providers, providerId, activeProvider };
};
