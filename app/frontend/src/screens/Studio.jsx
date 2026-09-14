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
        {tab === "data" && "Import media into a dataset draft, pre-process, and create."}
        {tab === "train" && "Start a run or inspect checkpoints and loss."}
        {tab === "models" && "Library models ready for Create."}
      </p>
      {tab === "data" && <Prepare />}
      {tab === "train" && <Train />}
      {tab === "models" && <ModelBrowser />}
    </div>
  );
}
