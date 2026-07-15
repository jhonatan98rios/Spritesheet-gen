import { ChatOpenAI } from "@langchain/openai";
import type { UsageMetadata } from "@langchain/core/messages";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export interface SpriteData {
  csv: string;
  palette: Record<string, string>;
}

export interface Attempt {
  messages: { role: string; content: string }[];
  rawOutput: string;
  error: string | null;
  usage: { inputTokens: number; outputTokens: number; totalTokens: number; costUSD: string } | null;
}

export interface SpriteResult {
  sprite: SpriteData | null;
  attempts: Attempt[];
  finalError: string | null;
}

// DeepSeek V4 Flash pricing per 1M tokens
const INPUT_PRICE_PER_1M = 0.14;
const OUTPUT_PRICE_PER_1M = 0.28;
const MAX_RETRIES = 3;

const SYSTEM_PROMPT = `You are a pixel art generator. You output ONLY valid JSON — no markdown, no explanation, no code fences. Your entire response must parse as JSON.

The JSON object has exactly two keys:

1. "csv" — a string containing EXACTLY 64 rows, each row containing EXACTLY 64 comma-separated integers from 0 to 9. Rows are joined by the escape sequence \\n (backslash-n, not a literal newline). No trailing newline. No spaces between commas and numbers.
   - 0 means transparent/background/empty space. Use it generously for empty areas around the subject.
   - Values 1-9 represent different colors (defined in the palette).
   - The sprite must be a recognizable representation of the user's description, centered in the 64x64 grid.
   - Use solid blocks of color (no dithering, no gradients). Think retro 8-bit pixel art.
   - The subject should occupy roughly 30-60% of the grid area, centered.

2. "palette" — an object with keys "0" through "9" mapping to CSS color strings (hex like "#FF8844" or rgba like "rgba(255,136,68,1)").
   - Key "0" MUST be a fully transparent color: "rgba(0,0,0,0)".
   - Keys "1" through "9" should be the colors used in the CSV, chosen to match the described subject.
   - Use distinct, vibrant colors so each index is clearly visible.
   - Every key "0" through "9" MUST be present exactly once.

CRITICAL — the "csv" value is a SINGLE-LINE JSON string. Use \\n escape sequences for row separators. The value must NOT contain literal newline characters. Example: "0,0,0,1,1,0\\\\n0,0,1,1,1,1,0\\\\n..."

CRITICAL CONSTRAINTS:
- The CSV must have EXACTLY 64 rows. Count them.
- Each row must have EXACTLY 64 comma-separated values.
- Use ONLY the digits 0-9 in the CSV.
- All 10 palette keys ("0" through "9") MUST be present.
- Output ONLY the JSON object. No markdown, no text.`;

function buildUserPrompt(description: string): string {
  return `Generate a 64x64 pixel art sprite of: ${description}`;
}

function buildRetryPrompt(description: string, error: string, previousOutput: string): string {
  return `Your previous response for "${description}" was invalid. The error was: ${error}

Your previous response was:
\`\`\`
${previousOutput.slice(0, 2000)}
\`\`\`

Please fix all issues and return ONLY valid JSON with "csv" and "palette" fields. Remember:
- "csv" must be a SINGLE-LINE JSON string using \\\\n for row separators (NOT literal newlines)
- EXACTLY 64 rows × 64 columns
- ALL palette keys "0" through "9" must be present`;
}

function sanitizeCsv(csv: string): string {
  const rows = csv.trim().split("\n");

  let normalized: string[];
  if (rows.length > 64) {
    normalized = rows.slice(0, 64);
  } else if (rows.length < 64) {
    const emptyRow = Array(64).fill("0").join(",");
    normalized = [...rows];
    while (normalized.length < 64) {
      normalized.push(emptyRow);
    }
  } else {
    normalized = rows;
  }

  return normalized
    .map((row) => {
      const cols = row.split(",");
      if (cols.length > 64) return cols.slice(0, 64).join(",");
      if (cols.length < 64) {
        return cols.concat(Array(64 - cols.length).fill("0")).join(",");
      }
      return row;
    })
    .join("\n");
}

function repairJsonWithLiteralNewlines(json: string): string {
  const csvKey = '"csv"';
  const paletteKey = '"palette"';

  const csvStart = json.indexOf(csvKey);
  const paletteStart = json.indexOf(paletteKey);
  if (csvStart === -1 || paletteStart === -1) return json;

  const colonIdx = json.indexOf(":", csvStart);
  if (colonIdx === -1 || colonIdx > paletteStart) return json;

  const valueOpen = json.indexOf('"', colonIdx + 1);
  if (valueOpen === -1 || valueOpen > paletteStart) return json;

  const rawMiddle = json.slice(valueOpen + 1, paletteStart);
  let closeIdx = rawMiddle.length - 1;
  while (closeIdx >= 0 && rawMiddle[closeIdx] !== '"') closeIdx--;
  if (closeIdx < 0) return json;

  let csvContent = rawMiddle.slice(0, closeIdx);

  csvContent = csvContent
    .replace(/\\/g, "\\\\")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");

  const before = json.slice(0, valueOpen + 1);
  const after = rawMiddle.slice(closeIdx);
  return before + csvContent + after + json.slice(paletteStart);
}

function parseJson(text: string): Record<string, unknown> {
  let json = text.trim();
  if (json.startsWith("```")) {
    json = json.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
  }

  try {
    return JSON.parse(json) as Record<string, unknown>;
  } catch {
    const repaired = repairJsonWithLiteralNewlines(json);
    return JSON.parse(repaired) as Record<string, unknown>;
  }
}

function validateAndExtract(parsed: Record<string, unknown>): SpriteData {
  if (typeof parsed.csv !== "string") {
    throw new Error("Response missing 'csv' field");
  }
  const csv = sanitizeCsv(parsed.csv);

  if (typeof parsed.palette !== "object" || parsed.palette === null) {
    throw new Error("Response missing 'palette' field");
  }
  const paletteObj = parsed.palette as Record<string, unknown>;
  const palette: Record<string, string> = {};
  for (let i = 0; i <= 9; i++) {
    const key = String(i);
    if (typeof paletteObj[key] !== "string") {
      throw new Error(`Palette missing key "${key}"`);
    }
    palette[key] = paletteObj[key];
  }

  return { csv, palette };
}

function computeCost(usage: UsageMetadata): string {
  const inputCost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_1M;
  const outputCost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M;
  const total = inputCost + outputCost;
  return total < 0.0001 ? "<$0.0001" : `$${total.toFixed(4)}`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractText(content: any): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((c) => (typeof c === "string" ? c : JSON.stringify(c))).join("");
  }
  return "";
}

export async function generateSprite(description: string): Promise<SpriteResult> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new Error("DEEPSEEK_API_KEY environment variable is not set");
  }

  const model = new ChatOpenAI({
    model: "deepseek-v4-flash",
    temperature: 0.7,
    maxTokens: 16384,
    configuration: {
      baseURL: "https://api.deepseek.com/v1",
      apiKey,
    },
  });

  const attempts: Attempt[] = [];
  let lastError: string | null = null;
  let sprite: SpriteData | null = null;

  for (let i = 0; i < MAX_RETRIES; i++) {
    const isRetry = i > 0;
    const userContent = isRetry
      ? buildRetryPrompt(description, lastError!, attempts[i - 1].rawOutput)
      : buildUserPrompt(description);

    const messages = [
      new SystemMessage(SYSTEM_PROMPT),
      new HumanMessage(userContent),
    ];

    const response = await model.invoke(messages);
    const rawOutput = extractText(response.content);
    const usageMeta = response.usage_metadata;

    const attemptMessages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ];

    let attemptError: string | null = null;
    let attemptUsage: Attempt["usage"] = null;

    if (usageMeta) {
      attemptUsage = {
        inputTokens: usageMeta.input_tokens,
        outputTokens: usageMeta.output_tokens,
        totalTokens: usageMeta.total_tokens,
        costUSD: computeCost(usageMeta),
      };
    }

    try {
      const parsed = parseJson(rawOutput);
      sprite = validateAndExtract(parsed);
      // Success — record attempt and break
      attempts.push({
        messages: attemptMessages,
        rawOutput: attemptOutputPreview(rawOutput),
        error: null,
        usage: attemptUsage,
      });
      break;
    } catch (err) {
      attemptError = err instanceof Error ? err.message : String(err);
      lastError = attemptError;
      attempts.push({
        messages: attemptMessages,
        rawOutput: attemptOutputPreview(rawOutput),
        error: attemptError,
        usage: attemptUsage,
      });
    }
  }

  return {
    sprite,
    attempts,
    finalError: sprite ? null : lastError,
  };
}

/** Truncate raw output for display — keep it readable */
function attemptOutputPreview(raw: string): string {
  if (raw.length <= 4000) return raw;
  return raw.slice(0, 2000) + `\n\n... [${raw.length - 4000} chars trimmed] ...\n\n` + raw.slice(-2000);
}
