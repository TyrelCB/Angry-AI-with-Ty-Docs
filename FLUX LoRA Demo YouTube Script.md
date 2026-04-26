# FLUX LoRA Demo YouTube Script

## Title

Training A Personal FLUX LoRA End To End And Testing It In ComfyUI

## Short Description

In this demo I walk through preparing a personal image dataset, fine-tuning a FLUX.1-dev LoRA, solving the ComfyUI compatibility issues, and comparing multiple checkpoints side by side.

## Opening Hook

Today I’m going to show you an end-to-end FLUX LoRA workflow that starts with a messy personal photo dataset and ends with a working ComfyUI comparison workflow. This is not a polished theory video. It is a real build, with the exact friction you hit when you actually try to do this on your own machine.

## Segment 1: What We Started With

On screen, show the repo root and the original dataset folder.

Voiceover:

I started by asking a simple question: is this dataset ready for LoRA fine-tuning? The answer was no. The dataset had mixed files, missing captions, EXIF rotation issues, and a bunch of images that were questionable for a subject LoRA.

Then I had the system clean it up instead of just talking about what should be done.

## Segment 2: The Dataset Audit

On screen:

- original dataset folder
- prepared dataset folder
- manifest

Voiceover:

The first pass found the usual problems. Some files were not usable for training, some images depended on rotation metadata, and there were no captions. A cleanup script was added to normalize images, rewrite them as JPGs, and split them into a conservative `train` folder and a `review` folder.

The result was thirteen train-ready images and nineteen review images.

That is small, but for a subject LoRA it is enough to do a solid first pass.

## Segment 3: Captions And Trigger Token

On screen:

- caption sidecar examples

Voiceover:

Next we generated caption sidecars. At first the default trigger token was a placeholder, and later I changed it to `Tyrel Barstow`. The captions were regenerated so every training example used the actual trigger phrase we wanted for prompting.

That detail matters, because if your captions and your actual prompt strategy drift apart, your results get worse fast.

## Segment 4: Converting The Dataset For Training

On screen:

- `build_flux_imagefolder_dataset.py`
- `diffusers-train/metadata.jsonl`

Voiceover:

After cleaning and captioning the dataset, it was rebuilt into a local `diffusers` imagefolder dataset with `metadata.jsonl`. That let the FLUX DreamBooth training script consume per-image captions cleanly instead of relying on one global instance prompt.

## Segment 5: Training Setup

On screen:

- `.venv-flux`
- `vendor/diffusers`
- `train_flux_lora.sh`

Voiceover:

Then the training environment was set up with a dedicated virtual environment, a local checkout of `diffusers`, and a shell script to launch the FLUX DreamBooth LoRA trainer.

This part had one of the first real-world snags. Hugging Face auth looked fine interactively, but it was not visible in non-interactive shells because the token export in `.bashrc` was below an early return. That had to be diagnosed and worked around.

That is exactly the kind of thing that never shows up in clean tutorial screenshots, but it absolutely shows up on real systems.

## Segment 6: Downloading The Base Model

On screen:

- `models/FLUX.1-dev-diffusers`

Voiceover:

The next step was downloading the full gated `FLUX.1-dev` diffusers snapshot. Not one safetensors file. The full model layout. That is important because the trainer expects the whole diffusers directory structure.

Once the full model was in place, training could actually start.

## Segment 7: The Actual Fine-Tune

On screen:

- training output directory
- checkpoints
- final LoRA file

Voiceover:

The run used the prepared train set, the `Tyrel Barstow person` trigger phrase, rank sixteen, alpha sixteen, and a target of five hundred steps.

The training completed successfully and took about six hours and fifty-three minutes wall-clock.

The important artifacts were the final LoRA weights and the intermediate checkpoints at steps three hundred, four hundred, and five hundred.

## Segment 8: Why Checkpoints Matter

On screen:

- checkpoint `300`
- checkpoint `400`
- checkpoint `500`

Voiceover:

This is one of the most important points in the whole workflow. The final weights are not automatically the best weights.

For subject LoRAs, the earlier checkpoint can often look more natural, while the later checkpoint can have stronger likeness but start to overfit or narrow the style range.

So instead of assuming the final export is best, we built the workflow to compare checkpoints directly.

## Segment 9: The ComfyUI Problem

On screen:

- ComfyUI error logs
- model dropdown mismatch

Voiceover:

Now here is where the workflow got messy in a useful way.

We tried loading the trained LoRA into ComfyUI, and it failed. The first issue was easy: the workflow referenced files ComfyUI could not see yet.

Then the deeper issue showed up. The FLUX base model had been downloaded in Hugging Face diffusers shard format, but the local ComfyUI loaders only wanted direct `.safetensors` files in the dropdowns. They would not accept the shard index JSON files.

That meant a normal workflow tweak was not enough.

## Segment 10: The Fix For ComfyUI

On screen:

- `merge_flux_shards_for_comfyui.py`
- merged transformer file
- merged T5 file

Voiceover:

The fix was to merge the FLUX transformer shards and the T5 shards into single ComfyUI-readable `.safetensors` files. After that, the workflow could point at merged model files instead of Hugging Face shard indexes.

Then separate LoRA entries were created for the step three hundred, four hundred, five hundred, and final weights so they could all be selected directly inside ComfyUI.

## Segment 11: The Comparison Workflow

On screen:

- `tyrel_barstow_flux_lora_compare_300_400_500_api.json`

Voiceover:

At that point, instead of testing one checkpoint at a time, a comparison workflow was created that runs the same prompt, same seed, same sampler, and same latent through three different LoRA checkpoints in one queue.

That is the cleanest way to evaluate checkpoint quality, because you are isolating the checkpoint as the only variable.

## Segment 12: What The Conversation Actually Looked Like

Voiceover:

This is worth saying out loud because it is part of the demo.

The process did not go:

- prepare data
- click train
- done

It went more like this:

- audit dataset
- clean and split it
- generate captions
- change the trigger token
- fix the training environment
- diagnose shell auth behavior
- wait for a huge model download
- complete a long training run
- hit ComfyUI loader errors
- inspect what ComfyUI would actually load
- merge FLUX shards into usable model files
- build a checkpoint comparison workflow

That is the real value of the workflow. Not just that it works, but that the failure modes are documented.

## Segment 13: Demo Close

On screen:

- generated outputs from `300`, `400`, and `500`

Voiceover:

So the finished result is not just a trained personal FLUX LoRA. It is a repeatable pipeline:

- clean the dataset
- caption it
- train it
- test it in Python
- make it work in ComfyUI
- compare checkpoints properly

If you are doing subject LoRAs, that last step matters more than people think. The best result is often not the final checkpoint. It is the one that best balances likeness and flexibility.

## Suggested B-Roll Checklist

- repo root
- raw dataset folder
- prepared `train/` and `review/` folders
- caption `.txt` files
- `metadata.jsonl`
- `train_flux_lora.sh`
- Hugging Face model directory
- training output directory
- checkpoint folders
- ComfyUI error logs
- merge script
- single workflow JSON
- comparison workflow JSON
- final three-image checkpoint comparison

## Suggested Closing Line

If you want the raw scripts and the exact workflow files, they are all in the repo. The important part is that this is not just a nice-looking result. It is a documented path from messy dataset to usable FLUX LoRA in ComfyUI.

## Variant: 3-5 Minute Version

### Fast Hook

I took a messy personal photo dataset, turned it into a FLUX LoRA training set, fine-tuned a model, hit real ComfyUI compatibility issues, fixed them, and ended with a workflow that compares multiple checkpoints in one run.

### Short Script

I started by checking whether the dataset was actually ready for LoRA training. It was not. The images needed cleanup, captions, EXIF rotation fixes, and a cleaner training split.

So the first thing I did was run a prep pipeline that normalized the images, split them into `train` and `review`, and generated caption sidecars. The final train set was only thirteen images, but that is enough for a first subject LoRA if the data is clean.

Then I rebuilt the dataset into a caption-aware `diffusers` format with `metadata.jsonl` and launched FLUX.1-dev LoRA training using the official DreamBooth path. That run took about six hours and fifty-three minutes and produced checkpoints at `300`, `400`, and `500`, plus a final LoRA.

But the interesting part came after training. When I moved into ComfyUI, the workflow did not just work. The local ComfyUI loaders would not accept the Hugging Face shard index files for the FLUX transformer and T5 encoder. They only wanted direct `.safetensors` files.

So instead of pretending that was fine, I fixed it properly. I merged the FLUX shards into single ComfyUI-readable weights, added separate LoRA entries for checkpoints `300`, `400`, and `500`, and built a comparison workflow that runs the same prompt through all three in one queue.

That is the real takeaway. The final checkpoint is not automatically the best one. For subject LoRAs, the earlier checkpoint can sometimes look more natural, while later ones can be more overfit. So the workflow is not just training. It is training plus checkpoint evaluation.

If you want the exact scripts, dataset prep flow, and ComfyUI setup, they are all documented in the repo.

### 3-5 Minute Shot List

- show original dataset
- show prepared `train/` and `review/`
- show captions with `Tyrel Barstow person`
- show training output and checkpoints
- show ComfyUI error
- show merge script
- show checkpoint comparison workflow
- show final `300` vs `400` vs `500` outputs

## Variant: 10-12 Minute Version

### Longer Hook

This is a full end-to-end FLUX LoRA workflow built on a real machine with real friction. I’m going to show the dataset cleanup, the actual training setup, the checkpoint outputs, the ComfyUI loader problems, and how I fixed them so I could compare multiple checkpoints side by side.

### Long Script

The starting point was a personal image dataset that looked usable at a glance, but it was not training-ready. There were mixed files, no captions, EXIF rotation problems, and a lot of images that were questionable for a tight subject LoRA.

So the first step was not training. It was dataset audit and cleanup. A prep script was added that normalized images, rewrote them as JPGs, applied EXIF rotation, and split the dataset into a conservative `train` set and a `review` set. That produced thirteen train-ready images and nineteen review images.

After that, caption sidecars were generated. Initially the trigger token was a placeholder, but I changed it to `Tyrel Barstow`, regenerated the captions, and verified that the old token was no longer present. That is a small step, but it matters because your training captions and your intended prompting strategy need to match.

The next step was converting the cleaned dataset into a format the FLUX DreamBooth LoRA script could use cleanly. That meant building a local imagefolder dataset with `metadata.jsonl` so the trainer could read per-image captions.

Then came the environment setup. A dedicated `.venv-flux` virtual environment was created, `diffusers` was cloned locally, and helper scripts were added for setup, model download, training, and testing.

This is where one of the real-world debugging moments happened. Hugging Face auth looked fine in the interactive terminal, but it was not visible to non-interactive shells because `.bashrc` returned early before the `HF_TOKEN` export line. That had to be diagnosed before the model download and training path would work reliably.

Once auth was available in the right shell context, the full gated `FLUX.1-dev` diffusers snapshot was downloaded. That is important because the trainer does not just want one safetensors file. It wants the full model layout.

Training used the prepared train set, the `Tyrel Barstow person` trigger phrase, a rank of sixteen, alpha sixteen, and a target of five hundred steps. The run completed successfully, took about six hours and fifty-three minutes, and produced checkpoints at `300`, `400`, and `500`, plus a final LoRA export.

That already gives you something useful, but the next point is the one I think people skip too often. The final checkpoint is not guaranteed to be the best checkpoint. Earlier checkpoints can be less overfit, more natural, and more flexible, even if the later checkpoint has slightly stronger identity lock-in.

So instead of just taking the final weights and calling it done, I moved into evaluation.

The first evaluation path was a simple Python script that loads the base FLUX model plus the LoRA and runs a prompt. That confirmed the LoRA artifact was usable. But for an actual visual checkpoint comparison, ComfyUI is more convenient.

That is where the second major debugging phase started.

The first attempt used a normal ComfyUI workflow with the FLUX base model and the trained LoRA. It failed because ComfyUI could not see the expected model files. After fixing that, a deeper compatibility issue appeared. The FLUX base model was in Hugging Face diffusers shard format, but the local ComfyUI loaders only wanted direct `.safetensors` files in their dropdowns. They would not accept the `.index.json` shard references.

At that point, changing a filename in the workflow was not enough. The actual model format had to be bridged.

So a dedicated merge script was added to consolidate the FLUX transformer shards into one ComfyUI-readable transformer file and the T5 shards into one ComfyUI-readable text encoder file. Those merged files were placed where ComfyUI expects them.

Then separate LoRA entries were created in the ComfyUI `loras` directory for:

- checkpoint `300`
- checkpoint `400`
- checkpoint `500`
- the final export

That made it possible to select each checkpoint independently.

After that, two ComfyUI workflows were created. The first one loads a single LoRA. The second one is the more interesting one: it takes the same prompt, same seed, same sampler, and same latent and runs them through the `300`, `400`, and `500` LoRA checkpoints in one queue.

That is the right way to compare checkpoints, because it isolates the checkpoint as the variable instead of mixing in prompt drift or random seed changes.

So the final state of the project is not just a trained subject LoRA. It is a full documented pipeline:

- audit dataset
- clean dataset
- caption dataset
- convert it for training
- fine-tune FLUX
- preserve intermediate checkpoints
- solve ComfyUI model-format issues
- compare checkpoints properly

And for me, that is the value of the demo. It shows the parts that usually get edited out: token mismatch, shell auth mismatch, model format mismatch, and workflow mismatch. Those are the things that actually make or break the experience when you try to reproduce a result on your own hardware.

### 10-12 Minute Shot List

- intro on goal of the demo
- original dataset inspection
- prepared dataset and manifest
- caption sidecars and token update
- diffusers training dataset build
- env setup scripts
- HF auth explanation
- model download folder
- training script and training outputs
- checkpoint discussion
- initial ComfyUI failure
- loader mismatch explanation
- shard merge script
- single workflow
- comparison workflow
- side-by-side results
- closing lesson on checkpoint selection

## Variant: Angry AI With Ty Version

### Tone Goal

More direct. Less tutorial-polished. More emphasis on what broke, why it broke, and why most “easy workflow” demos skip the hard parts.

### Script

Today I’m going to show you what it actually looks like to train a personal FLUX LoRA instead of pretending this is a three-click workflow.

Because it is not.

It started with a simple question: is this dataset ready for LoRA fine-tuning?

No.

The dataset had the usual junk people love to ignore in tutorials:

- mixed files
- missing captions
- EXIF rotation problems
- weak samples
- inconsistent naming

So the first move was not training. It was cleanup.

A prep script was used to normalize the images, rewrite them as JPGs, fix orientation, and split the data into a `train` folder and a `review` folder. After cleanup, the actual trainable set was thirteen images.

And before somebody says thirteen is too small, for a subject LoRA, thirteen clean images can beat fifty sloppy ones every time.

Then captions were generated. The trigger token started as a placeholder, and later I changed it to `Tyrel Barstow` because the whole point is to train against the thing you actually intend to prompt with.

That sounds obvious, but this is exactly the kind of mismatch that quietly ruins results while people blame the model.

After that, the dataset was rebuilt into a `diffusers` imagefolder dataset with `metadata.jsonl`, because if you want a clean training path, you want the trainer consuming real per-image captions, not a vague hand-wave prompt.

Then came environment setup. Dedicated virtual environment. Local `diffusers` checkout. Training scripts. Download scripts. Test scripts. All normal so far.

Then real life showed up.

Hugging Face auth looked fine in the terminal, but the token was invisible in non-interactive shells because `.bashrc` returned early before the export line. So the system looked authenticated until it actually had to do work.

That right there is the difference between a fake tutorial and a real workflow.

Once that was sorted out, the full gated `FLUX.1-dev` diffusers snapshot was downloaded and training finally started.

The run used the cleaned train set, the `Tyrel Barstow person` trigger phrase, rank sixteen, alpha sixteen, and five hundred steps. It ran for about six hours and fifty-three minutes and produced checkpoints at `300`, `400`, and `500`, plus the final LoRA weights.

Now here is the part a lot of people still get wrong. The final checkpoint is not automatically the best checkpoint.

Sometimes `300` is cleaner.
Sometimes `400` is the sweet spot.
Sometimes `500` has stronger likeness but is already drifting into overfit territory.

So if your workflow only tests the final weights, your workflow is incomplete.

Then I moved into ComfyUI, and that is where the next mess started.

The first workflow failed because ComfyUI could not see the referenced files. Easy fix.

Then the actual problem showed up. The FLUX model was in Hugging Face diffusers shard format, but the ComfyUI loaders on this machine only wanted direct `.safetensors` files in their dropdowns. They would not accept the shard index JSON files.

So no, this was not a matter of “just load the model.”

The fix was to merge the FLUX transformer shards and the T5 shards into single ComfyUI-readable files. Then separate LoRA entries were added for checkpoint `300`, `400`, `500`, and final.

And then, instead of doing the lazy thing and testing one checkpoint at a time with slightly different prompts and seeds, a proper comparison workflow was built. Same prompt. Same seed. Same sampler. Same latent. Three checkpoints in one queue.

That is how you actually compare checkpoints.

So the finished output here is not just a personal FLUX LoRA. It is a real pipeline:

- audit the data
- clean the data
- caption the data
- convert the data
- train the model
- save checkpoints
- fix the ComfyUI format mismatch
- compare checkpoints properly

And that is what I want people to take away from this demo.

The hard part is not just getting a model to run once.

The hard part is building a workflow that survives the boring failures:

- shell auth weirdness
- dataset quality issues
- token mismatch
- model format mismatch
- workflow mismatch

That is the difference between a cool screenshot and a reproducible system.

### Angry AI Shot List

- original messy dataset
- prepared dataset split
- caption examples with `Tyrel Barstow person`
- training scripts
- output checkpoints
- failed ComfyUI validation
- merged FLUX files
- multi-checkpoint comparison workflow
- final side-by-side result grid

### Angry AI Closing Line

If somebody shows you a perfect FLUX LoRA workflow with no friction, they are either skipping the hard parts or they already solved them off camera. This version keeps the hard parts in, because that is where the actual value is.
