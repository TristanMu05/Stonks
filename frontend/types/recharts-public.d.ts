declare module "recharts" {
  export const usePlotArea: () =>
    | { x: number; y: number; width: number; height: number }
    | undefined;
}
