# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from pathlib import Path
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive


def interactive_single_clip(fbx_path, fbx_name, root_joint="Hips", fps=60):
    # source fbx file path
    fbx_file = Path(fbx_path) / fbx_name

    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file.as_posix(),
        root_joint=root_joint,
        fps=fps
    )

    # save motion in npy format
    outfile = Path(fbx_path) / (fbx_name.split(".")[0] + ".npy")
    motion.to_file(outfile.as_posix())

    # visualize motion
    plot_skeleton_motion_interactive(motion)

    return


def batch_convert(fbx_path, root_joint="Hips", fps=60):
    # source fbx file path
    fbx_path = "data/CMU_fbx_sub16/"

    all_files = sorted(Path(fbx_path).glob("*.fbx"))
    
    for fbx_file in all_files:
        # import fbx file - make sure to provide a valid joint name for root_joint
        motion = SkeletonMotion.from_fbx(
            fbx_file_path=fbx_file.as_posix(),
            root_joint=root_joint,
            fps=fps
        )

        # save motion in npy format
        fbx_name = fbx_file.name
        outfile = Path(fbx_path) / (fbx_name.split(".")[0] + ".npy")
        motion.to_file(outfile.as_posix())

        print(f"{outfile.as_posix()} saved to disk")
    return


if __name__ == "__main__":
    # specify path and name. input data location could be different
    interactive_single_clip("data/CMU_fbx_sub16/", "16_11.fbx")
    # interactive_single_clip("data/", "07_01_cmu.fbx")
    
    # provide path only
    # batch_convert("data/CMU_fbx_sub16/")