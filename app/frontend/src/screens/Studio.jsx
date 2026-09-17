import React from "react";
import Prepare from "./Prepare.jsx";
import Train from "./Train.jsx";
import ModelBrowser from "./ModelBrowser.jsx";
import { useApp } from "../state.jsx";

export default function Studio() {
  const { prepareTab } = useApp();
  const tab = ["data", "train", "models"].includes(prepareTab) ? prepareTab : "data";

  return (
    <div className="col">
      <p className="hint mb-0">
        {tab === "data" && "Point a dataset at your image folders, choose framing and augmentation, and train on it. Nothing is copied."}
        {tab === "train" && "Start a run or inspect checkpoints and loss."}
        {tab === "models" && "Library models ready for Create."}
      </p>
      {tab === "data" && <Prepare />}
      {tab === "train" && <Train />}
      {tab === "models" && <ModelBrowser />}
    </div>
  );
}
