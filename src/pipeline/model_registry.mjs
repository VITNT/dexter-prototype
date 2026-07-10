export async function loadCatalog(catalogUrl = "data/catalog/sign-catalog.example.json") {
  const response = await fetch(catalogUrl);
  if (!response.ok) {
    throw new Error(`Unable to load sign catalog: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export function getLanguageConfig(catalog, languageId) {
  const language = catalog?.languages?.find(entry => entry.id === languageId);
  if (!language) {
    throw new Error(`Unknown sign language: ${languageId}`);
  }
  return language;
}

export function getModelConfig(catalog, languageId, modelId) {
  const language = getLanguageConfig(catalog, languageId);
  const models = language.models || [];

  if (models.length === 0) {
    throw new Error(`No models registered for language: ${languageId}`);
  }

  if (!modelId) return withLandmarkLayout(catalog, models[0]);

  const model = models.find(entry => entry.id === modelId);
  if (!model) {
    throw new Error(`Model ${modelId} is not registered for language: ${languageId}`);
  }
  return withLandmarkLayout(catalog, model);
}

export async function loadSequenceModel({ tf, catalog, languageId, modelId }) {
  if (!tf?.loadLayersModel) {
    throw new Error("TensorFlow.js must be provided as { tf }.");
  }

  const modelConfig = getModelConfig(catalog, languageId, modelId);
  const [labels, model, metadata] = await Promise.all([
    loadJson(modelConfig.labelsPath),
    tf.loadLayersModel(modelConfig.modelPath),
    modelConfig.metadataPath ? loadJson(modelConfig.metadataPath) : Promise.resolve(null)
  ]);

  const outputCount = model?.outputs?.[0]?.shape?.at(-1);
  if (Number.isFinite(outputCount) && outputCount !== labels.length) {
    throw new Error(
      `Model ${modelConfig.id} outputs ${outputCount} classes, but ${modelConfig.labelsPath} contains ${labels.length} labels.`
    );
  }

  return {
    model,
    labels,
    metadata,
    config: modelConfig
  };
}

export function findSign(catalog, languageId, labelOrGloss) {
  const language = getLanguageConfig(catalog, languageId);
  const target = normalizeLabel(labelOrGloss);

  return (language.signs || []).find(sign => (
    normalizeLabel(sign.label) === target ||
    normalizeLabel(sign.gloss) === target ||
    normalizeLabel(sign.id) === target
  )) || null;
}

export function translateLabel(catalog, languageId, labelOrGloss, targetLanguage) {
  const language = getLanguageConfig(catalog, languageId);
  const sign = findSign(catalog, languageId, labelOrGloss);
  const fallbackTarget = targetLanguage || language.defaultTargetLanguage || "en";

  if (!sign) {
    return {
      text: String(labelOrGloss),
      source: "label_fallback"
    };
  }

  return {
    text: sign.translations?.[fallbackTarget] || sign.gloss || sign.label,
    source: sign.translations?.[fallbackTarget] ? "catalog" : "gloss_fallback",
    sign
  };
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Unable to load JSON from ${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function normalizeLabel(value) {
  return String(value || "").trim().toLowerCase();
}

function withLandmarkLayout(catalog, modelConfig) {
  const layout = catalog?.landmarkLayouts?.[modelConfig.landmarkLayout];
  return {
    ...modelConfig,
    frameWidth: layout?.frameWidth,
    schemaVersion: modelConfig.schemaVersion || layout?.schemaVersion,
    modelFrameWidth: modelConfig.modelFrameWidth || layout?.modelFrameWidth || layout?.frameWidth
  };
}
