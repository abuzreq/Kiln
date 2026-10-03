import React from "react";
import Prepare from "./Prepare.jsx";
import Train from "./Train.jsx";
import ModelBrowser from "./ModelBrowser.jsx";
import { useApp } from "../state.jsx";
import OpenFolder from "../components/OpenFolder.jsx";

export default function Studio() {
  const { prepareTab } = useApp();
  const tab = ["data", "train", "models"].includes(prepareTab) ? prepareTab : "data";

  return (
    <div className="col">
      <div className="row between center gap-2 wrap">
        <p className="hint mb-0">
          {tab === "data" && "Point a dataset at your image folders, choose framing and augmentation, and train on it. Nothing is copied."}
          {tab === "train" && "Start a run or inspect checkpoints and loss."}
          {tab === "models" && "Library models ready for Create."}
        </p>
        {tab === "data" && <OpenFolder folder="datasets" label="Open datasets folder" />}
        {tab === "models" && <OpenFolder folder="models" label="Open models folder" />}
      </div>
      {tab === "data" && <Prepare />}
      {tab === "train" && <Train />}
      {tab === "models" && <ModelBrowser />}
    </div>
  );
}
