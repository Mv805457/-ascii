# ASCIIfy Core Mathematics Explained

The core of `asciify` isn't standard image processing—it is pure linear algebra functioning directly on pixel intensity matrices. Below is a breakdown of the mathematical transformations applied to every single frame.


## 1. Grayscale Conversion (Dimension Reduction)
**What:** The input image starts as an RGB NumPy tensor of shape `(Height, Width, 3)`. We compress this into a mathematically manageable 2D matrix of shape `(H, W)`.
**How:** We use the standard Rec. 601 (or Rec. 709) luminance weights:
   L = 0.2989 * R + 0.5870 * G + 0.1140 * B
**Why:** Most linear algebraic transformations (like SVD) are designed for 2-D matrices, not 3-D tensors. We extract purely the structural "brightness", discarding color data which is later overlaid mathematically.


## 2. Singular Value Decomposition (SVD)
**What:** SVD factors our real image matrix `A` into three distinct matrices: `U * S * V^T`. 
**How:** 
   1. `U` (Left Singular Vectors): Captures the vertical column patterns.
   2. `S` (Singular Values): A diagonal matrix of decreasing "weights". It tells us how important each pattern in `U` and `V^T` is. 
   3. `V^T` (Right Singular Vectors): Captures the horizontal row patterns.
   
We perform **Low-Rank Approximation** (SVD Rank). By taking only the largest `k` singular values from `S`, and the corresponding first `k` columns of `U` and `k` rows of `V`:
   A_approx = U_k * S_k * V_k^T
**Why:** A high-resolution image might have a mathematical rank of 1000. By setting the rank to 30, we forcefully delete all the small, noisy mathematical data ("high-frequency noise"). This results in a cleaner, smoother image, which translates exceptionally well to blocky ASCII rendering.


## 3. Sobel Edge Detection (Vectorized Convolution)
**What:** We need to find purely the contours and shapes within the matrix. We do this by calculating the mathematical gradient of the 2D matrix, measuring how fast brightness changes across neighboring pixels.
**How:** We define two 3x3 kernels (small matrices):
   Gx = [[-1,  0,  1],      Gy = [[-1, -2, -1],
         [-2,  0,  2],            [ 0,  0,  0],
         [-1,  0,  1]]            [ 1,  2,  1]]

These are essentially partial derivatives. We slide them over our matrix:
   * `gx` = Convolution(Matrix, Gx)  -> Returns the horizontal gradient
   * `gy` = Convolution(Matrix, Gy)  -> Returns the vertical gradient

**Edge Magnitude:**  
Calculated using the Euclidean norm (Pythagorean theorem): 
   Magnitude = sqrt(gx² + gy²)
This tells us *how strong* the edge is.

**Edge Direction:**
Calculated using trigonometry:
   Angle = arctan2(gy, gx)
This reveals the exact angle the gradient is pointing. We segment this angle mathematically into 4 "buckets" corresponding to `|`, `\`, `_`, and `/`.
**Why:** Without edge detection, an ASCII image is just a blurry mess. By computationally identifying where the "derivatives are highest" (areas changing color abruptly), we outline important structures, making faces and objects pop.


## 4. Vectorized Sliding Windows (The Optimization)
**What & How:** Running a python `for` loop over 800,000 pixels is extremely slow. We use NumPy's `lib.stride_tricks.as_strided` memory mapping. Instead of looping, we reorganize memory pointers to represent the image as overlapping 3x3 blocks, moving the convolution purely to C-level hardware calculations via Einstein Summation Notation (`np.einsum('ijk,k->ij')`).
**Why:** Makes edge detection thousands of times faster, enabling Real-Time 30+ FPS Native Webcam capture.


## 5. Matrix Scaling & ASCII Indexing
**What:** Combining SVD and Sobel yields a chaotic float64 matrix with values ranging wildly.
**How:**
   Normalization = (Matrix - Matrix.min()) / (Matrix.max() - Matrix.min()) 
This scales everything precisely from `0.0` to `1.0`. We multiply this by the number of characters in our ASCII ramp (e.g. 70 characters), floor the result, and apply standard array-indexing.
   Indices = (Normalized * (Num_Chars - 1)).astype(int)
**Why:** Binds our abstract linear algebra models precisely to our visual string/character representations, ensuring the darkest mathematical parts get drawn as ` ` and the brightest parts hit `@`.
