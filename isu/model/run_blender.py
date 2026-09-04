from pathlib import Path
import signal
import os
import subprocess
import time
from isu.config import BLENDER_APP_PATH, BLENDER_FILE_PATH, BLENDER_SCRIPT_PATH

def run_blender(json_path: str, 
                image_output_dir:str,
                depth_output_dir: str = None,
                seg_output_dir: str = None,
                canny_output_dir: str = None,
                instance_seg_output_dir: str = None,
                blender_exe=BLENDER_APP_PATH,      
                blend_file=BLENDER_FILE_PATH,
                blender_python_script=BLENDER_SCRIPT_PATH,
                max_retries: int = 3,
                timeout_sec: int = 100
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    blender_python_script = (project_root / blender_python_script).resolve()
    json_path = Path(json_path).resolve()
    image_output_dir = Path(image_output_dir).resolve()
    image_path = image_output_dir / (json_path.stem + "_sim.png")
    depth_path = (
        Path(depth_output_dir).resolve() / "exr" / (json_path.stem + "_depth.exr")
        if depth_output_dir else None
    )
    seg_path = (
        Path(seg_output_dir).resolve() / (json_path.stem + "_seg.png")
        if seg_output_dir else None
    )
    canny_path = (
        Path(canny_output_dir).resolve() / (json_path.stem + "_canny.png")
        if canny_output_dir else None
    )
    instance_seg_path = (
        Path(instance_seg_output_dir).resolve() / (json_path.stem + "_instance_seg.png")
        if instance_seg_output_dir else None
    )
    log_path = project_root / f"isu/blender/blender_simulation_{Path(json_path).parent.parent.parent.name}.log"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    for output_path in (depth_path, seg_path, canny_path, instance_seg_path):
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

    # blender --background your_scene.blend --python blender_driver.py -- input.json output.png
    def build_cmd(force_cpu=False):
        cmd = [
            str(blender_exe),
            "--background",
            "--factory-startup",
            "--verbose", "3",
            str(blend_file),
            "--python", str(blender_python_script),
            "--",
            str(json_path),
            str(image_path),
        ]
        if depth_path is not None:
            cmd.append(str(depth_path))
        if seg_path is not None:
            cmd.extend(["--seg-path", str(seg_path)])
        if canny_path is not None:
            cmd.extend(["--canny-path", str(canny_path)])
        if instance_seg_path is not None:
            cmd.extend(["--instance-seg-path", str(instance_seg_path)])
        if force_cpu:
            cmd.append("--force-cpu")  # your driver_rgb.py should obey this
        return cmd
    
    # --- main retry loop ---
    for attempt in range(1, max_retries + 1):
        force_cpu = (attempt > 1)  # 1st: GPU, 2nd: CPU fallback
        cmd = build_cmd(force_cpu)
        # print("cmd: ", cmd)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== Attempt {attempt}/{max_retries} (CPU={force_cpu}) ===\n")
            log.write(" ".join(cmd) + "\n")

            # Start Blender as a separate process group so we can kill it
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log,
                stderr=log,
                text=True,
                start_new_session=True
            )

            try:
                #record start time
                start_time = time.time()
                proc.wait(timeout=timeout_sec)
                #record end time
                end_time = time.time()
                elapsed_time = end_time - start_time
                log.write(f"\n[Completed in {elapsed_time:.2f}s]\n")
            except subprocess.TimeoutExpired:
                log.write(f"\n[Timeout after {timeout_sec}s] Killing Blender...\n")
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(1)
                os.killpg(proc.pid, signal.SIGKILL)
                rc = -9
            else:
                rc = proc.returncode

            # start_time = time.time()
            # elapsed_time = 0

            # while elapsed_time < timeout_sec:
            #     if os.path.exists(image_path):
            #         print(f"[INFO] Output file detected: {image_path}")
            #         break
            #     if proc.poll() is not None:  # Blender already exited
            #         break
            #     time.sleep(0.1)  
            #     elapsed_time = time.time() - start_time

            # log.write(f"\n[Completed in {elapsed_time:.2f}s]\n")
            # proc.terminate()

            # --- check results ---
            if rc == 0 and image_path.exists():
                return str(image_path)  # success

            # --- failed ---
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[Fail] Return code: {rc}, Image exists: {image_path.exists()}\n")

        # Kill any straggler just in case
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass

        # Retry (if not last)
        if attempt < max_retries:
            time.sleep(3)
            continue

        raise RuntimeError(
            f"Blender failed after {max_retries} attempts. "
            f"See log: {log_path}"
        )