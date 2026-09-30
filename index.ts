import type {
  ExtensionAPI,
  ModelChangeEntry,
  ModelRegistry,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import {
  clampThinkingLevel,
  createAssistantMessageEventStream,
  isContextOverflow,
  hasApi,
  type Api,
  type AssistantMessage,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";

const OPENAI_FAST_API = "openai-fast-responses";
const OPENAI_API = "openai-responses";
const OPENAI_FAST_PROVIDER = "openai-fast";
const OPENAI_PROVIDER = "openai";
const PLACEHOLDER_API_KEY = "__openai_fast_reuses_openai_auth__";
const OPENAI_FAST_MODEL_IDS = new Set([
  "gpt-6.1-sol",
  "gpt-6-astra",
  "gpt-6-luna",
  "gpt-6-sol",
  "gpt-5.6-luna",
  "gpt-5.6-terra",
  "gpt-5.6-sol",
  "gpt-5.5",
]);

type ExtensionDiagnostic = {
  type: "warning" | "error";
  code: "no-fast-models" | "no-model-base-url";
  message: string;
};

type Result<T> = { ok: true; value: T } | { ok: false; diagnostic: ExtensionDiagnostic };
type OpenAIApi = typeof OPENAI_API;

function getOpenAIFastModels(openAIModels: readonly Model<OpenAIApi>[]): ProviderModelConfig[] {
  return openAIModels
    .filter((model) => OPENAI_FAST_MODEL_IDS.has(model.id))
    .map((model): ProviderModelConfig => {
      const config: ProviderModelConfig = {
        id: model.id,
        name: model.name,
        baseUrl: model.baseUrl,
        reasoning: model.reasoning,
        input: model.input,
        cost: model.cost,
        contextWindow: model.contextWindow,
        maxTokens: model.maxTokens,
      };
      if (model.thinkingLevelMap !== undefined) {
        config.thinkingLevelMap = model.thinkingLevelMap;
      }
      if (model.headers !== undefined) {
        config.headers = model.headers;
      }
      if (model.compat !== undefined) {
        config.compat = model.compat;
      }
      return config;
    });
}

function getFastProviderBaseUrl(openAIFastModels: readonly ProviderModelConfig[]): Result<string> {
  if (openAIFastModels.length === 0) {
    return {
      ok: false,
      diagnostic: {
        type: "error",
        code: "no-fast-models",
        message: `No models available for ${OPENAI_FAST_PROVIDER}. The provider will not be registered.`,
      },
    };
  }

  const baseUrl = openAIFastModels.find((model) => model.baseUrl)?.baseUrl;
  if (!baseUrl) {
    return {
      ok: false,
      diagnostic: {
        type: "error",
        code: "no-model-base-url",
        message: `No base URL found for any ${OPENAI_FAST_PROVIDER} model. The provider will not be registered.`,
      },
    };
  }

  return { ok: true, value: baseUrl };
}

function endWithCanonicalError(
  stream: ReturnType<typeof createAssistantMessageEventStream>,
  modelId: string,
  errorMessage: string,
  options?: SimpleStreamOptions,
): void {
  const message: AssistantMessage = {
    role: "assistant",
    content: [],
    api: OPENAI_API,
    provider: OPENAI_PROVIDER,
    model: modelId,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: options?.signal?.aborted ? "aborted" : "error",
    errorMessage,
    timestamp: Date.now(),
  };
  stream.push({
    type: "error",
    reason: message.stopReason === "aborted" ? "aborted" : "error",
    error: message,
  });
  stream.end(message);
}

function streamSimpleOpenAIFast(
  modelRegistry: ModelRegistry | undefined,
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
) {
  const outer = createAssistantMessageEventStream();

  const streamTask = (async () => {
    if (!modelRegistry) {
      endWithCanonicalError(
        outer,
        model.id,
        `${OPENAI_FAST_PROVIDER} session is not initialized.`,
        options,
      );
      return;
    }

    const openAIModel = modelRegistry.find(OPENAI_PROVIDER, model.id);
    if (!openAIModel || !hasApi(openAIModel, OPENAI_API)) {
      endWithCanonicalError(
        outer,
        model.id,
        `Underlying ${OPENAI_PROVIDER} Responses model not found for ${model.id}.`,
        options,
      );
      return;
    }

    const clampedReasoning = options?.reasoning
      ? clampThinkingLevel(openAIModel, options.reasoning)
      : undefined;
    // Resolve credentials and provider overrides through the native OpenAI route.
    // Never forward the fast provider's placeholder key or resolved auth headers/env.
    const { apiKey: _apiKey, headers: _headers, env: _env, ...requestOptions } = options ?? {};
    const inner = modelRegistry.stream(openAIModel, context, {
      ...requestOptions,
      ...(clampedReasoning && clampedReasoning !== "off"
        ? { reasoningEffort: clampedReasoning }
        : {}),
      serviceTier: "priority",
    });

    for await (const event of inner) {
      if (event.type === "error" && isContextOverflow(event.error, model.contextWindow)) {
        outer.push({
          ...event,
          error: {
            ...event.error,
            provider: OPENAI_FAST_PROVIDER,
            model: model.id,
          },
        });
      } else {
        outer.push(event);
      }
    }
    outer.end();
  })();
  streamTask.catch((error: unknown) => {
    endWithCanonicalError(
      outer,
      model.id,
      error instanceof Error ? error.message : String(error),
      options,
    );
  });

  return outer;
}

export default function (pi: ExtensionAPI) {
  const openAIModels = getBuiltinModels(OPENAI_PROVIDER);
  const openAIFastModels = getOpenAIFastModels(openAIModels);
  const diagnostics: ExtensionDiagnostic[] = [];
  const baseUrl = getFastProviderBaseUrl(openAIFastModels);
  let modelRegistry: ModelRegistry | undefined;
  let providerRegistered = false;

  if (!baseUrl.ok) {
    diagnostics.push(baseUrl.diagnostic);
  } else {
    pi.registerProvider(OPENAI_FAST_PROVIDER, {
      name: "OpenAI Fast",
      baseUrl: baseUrl.value,
      apiKey: PLACEHOLDER_API_KEY,
      api: OPENAI_FAST_API,
      models: openAIFastModels,
      streamSimple: (model, context, options) =>
        streamSimpleOpenAIFast(modelRegistry, model, context, options),
    });
    providerRegistered = true;
  }

  pi.on("session_start", async (_event, ctx) => {
    modelRegistry = ctx.modelRegistry;
    for (const diagnostic of diagnostics.splice(0)) {
      if (ctx.hasUI) {
        ctx.ui.notify(diagnostic.message, diagnostic.type);
      } else if (diagnostic.type === "error") {
        console.error(`[${OPENAI_FAST_PROVIDER}] ${diagnostic.message}`);
      } else {
        console.warn(`[${OPENAI_FAST_PROVIDER}] ${diagnostic.message}`);
      }
    }
    if (!providerRegistered) {
      return;
    }

    const latestModelChange = ctx.sessionManager
      .getBranch()
      .findLast((entry): entry is ModelChangeEntry => entry.type === "model_change");

    if (latestModelChange?.provider !== OPENAI_FAST_PROVIDER) {
      return;
    }

    const { modelId } = latestModelChange;
    if (ctx.model?.provider === OPENAI_FAST_PROVIDER && ctx.model.id === modelId) {
      return;
    }

    const fastModel = ctx.modelRegistry.find(OPENAI_FAST_PROVIDER, modelId);
    if (fastModel) {
      await pi.setModel(fastModel);
    }
  });
}
