/** Browser-storage keys used by the frontend. Keep persisted UI state centralized. */
export const MEMORIA_SETTINGS_ADVANCED_MODE = "memoria.settings.advanced-mode";
export const MEMORIA_ACTIVE_INTENT_PRESET = "memoria.settings.active-intent-preset";
export const MEMORIA_USER_SETTINGS = "memoria.settings.preferences";

export const LOCAL_STORAGE_KEYS = {
  MEMORIA_SETTINGS_ADVANCED_MODE,
  MEMORIA_ACTIVE_INTENT_PRESET,
  MEMORIA_USER_SETTINGS,
} as const;

/** Photo blob-object URLs retained by the shared LRU pool per browser tab. */
export const BLOB_POOL_MAX_ENTRIES = 50;
