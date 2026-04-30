export type RecognizeResult = {
  digit: number
  confidence: number
}

export function recognizeDigit(
  canvas: HTMLCanvasElement
): Promise<RecognizeResult | null>
