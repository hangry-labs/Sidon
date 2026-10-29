# Third-party notices

Our contributions are licensed under Apache-2.0 in the root [LICENSE](LICENSE).
The following third-party grants and notices remain in effect:

| Material | License and preserved notice |
|---|---|
| Original Sidon project attribution | MIT; copyright (c) 2025 sarulab-speech, in [licenses/Sidon-upstream-MIT.txt](licenses/Sidon-upstream-MIT.txt) |
| Selected ClearerVoice-Studio MossFormer2 architecture | Apache-2.0; [licenses/ClearerVoice-Apache-2.0.txt](licenses/ClearerVoice-Apache-2.0.txt) |
| SpeechBrain-derived dual-path model portions | Apache-2.0; [licenses/SpeechBrain-Apache-2.0.txt](licenses/SpeechBrain-Apache-2.0.txt) |
| FLASH-pytorch-derived model portions | MIT; copyright (c) 2022 Phil Wang, in [licenses/FLASH-MIT.txt](licenses/FLASH-MIT.txt) |
| Pinned MossFormer2 SS weights, downloaded separately | Publisher declares Apache-2.0; preserved [model card](licenses/MossFormer2-MODEL-CARD.md) |
| Pinned Sidon restoration exports | Publisher declares MIT; preserved [model card](licenses/Sidon-MODEL-CARD.md) and original Sidon MIT grant |

The exact imported source revision, file hashes and URLs are recorded in
[provenance.json](src/sidon/inference/_mossformer2/provenance.json). Local changes
and the model asset's revision/hash are described in [RUNTIME_NOTICES.md](RUNTIME_NOTICES.md).
The notices and license files accompany the built distributions.

Dependencies and bundled native libraries retain their own terms. Their top-level
metadata is recorded in [the dependency inventory](docs/runtime-dependency-licenses.tsv);
it is a summary. Complete collected binary notices are in each image at
`/app/licenses/runtime`, with matching source provision in the `sources` target.


## Studio additions

The UI shell, GPU monitor and image/source publication helpers are adapted from
Hangry Labs faster-qwen3-tts-studio. The original MIT notice is retained in
`licenses/faster-qwen-upstream-MIT.txt`. Lucide icon font notices are distributed
in `src/sidon/web/static/vendor/lucide/LICENSE`. Shared Hangry Labs branding is
used with the project owner's authorization.

## Supplemental notices for installed wheels

| Package | Bundled component | Notice |
| --- | --- | --- |
| torch 2.10.0+cu128 | Intel oneMKL 2024.2.0 | Intel Simplified Software License, reproduced below. |

The pinned PyTorch wheel omits this bundled component's complete notice.
Build provenance: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/.ci/docker/common/install_mkl.sh
License source: https://cdrdv2-public.intel.com/749362/intel-simplified-license-software-october-2022.pdf

### Intel oneMKL license text

```text
Intel Simplified Software License (Version October 2022)

Use and Redistribution.  You may use and redistribute the software, which is provided in binary form only, (the “Software”), without modification, provided the following conditions are met:

* Redistributions must reproduce the above copyright notice and these terms of use in the Software and in the documentation and/or other materials provided with the distribution.
* Neither the name of Intel nor the names of its suppliers may be used to endorse or promote products derived from this Software without specific prior written permission.
* No reverse engineering, decompilation, or disassembly of the Software is permitted, nor any modification or alteration of the Software or its operation at any time, including during execution.

No other licenses.  Except as provided in the preceding section, Intel grants no licenses or other rights by implication, estoppel or otherwise to, patent, copyright, trademark, trade name, service mark or other intellectual property licenses or rights of Intel.

Third party software.  “Third Party Software” means the files (if any) listed in the “third-party-software.txt” or other similarly-named text file that may be included with the Software. Third Party Software, even if included with the distribution of the Software, may be governed by separate license terms, including without limitation, third party license terms, open source software notices and terms, and/or other Intel software license terms. These separate license terms solely govern Your use of the Third Party Software.

DISCLAIMER.  THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT ARE DISCLAIMED. THIS SOFTWARE IS NOT INTENDED FOR USE IN SYSTEMS OR APPLICATIONS WHERE FAILURE OF THE SOFTWARE MAY CAUSE PERSONAL INJURY OR DEATH AND YOU AGREE THAT YOU ARE FULLY RESPONSIBLE FOR ANY CLAIMS, COSTS, DAMAGES, EXPENSES, AND ATTORNEYS’ FEES ARISING OUT OF ANY SUCH USE, EVEN IF ANY CLAIM ALLEGES THAT INTEL WAS NEGLIGENT REGARDING THE DESIGN OR MANUFACTURE OF THE SOFTWARE.

LIMITATION OF LIABILITY. IN NO EVENT WILL INTEL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

No support.  Intel may make changes to the Software, at any time without notice, and is not obligated to support, update or provide training for the Software.

Termination. Your right to use the Software is terminated in the event of your breach of this license.

Feedback.  Should you provide Intel with comments, modifications, corrections, enhancements or other input (“Feedback”) related to the Software, Intel will be free to use, disclose, reproduce, license or otherwise distribute or exploit the Feedback in its sole discretion without any obligations or restrictions of any kind, including without limitation, intellectual property rights or licensing obligations.

Compliance with laws.  You agree to comply with all relevant laws and regulations governing your use, transfer, import or export (or prohibition thereof) of the Software.

Governing law.  All disputes will be governed by the laws of the United States of America and the State of Delaware without reference to conflict of law principles and subject to the exclusive jurisdiction of the state or federal courts sitting in the State of Delaware, and each party agrees that it submits to the personal jurisdiction and venue of those courts and waives any objections. THE UNITED NATIONS CONVENTION ON CONTRACTS FOR THE INTERNATIONAL SALE OF GOODS (1980) IS SPECIFICALLY EXCLUDED AND WILL NOT APPLY TO THE SOFTWARE.
```
