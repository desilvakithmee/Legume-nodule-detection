# =============================================================================
# Estimate total nodule area from YOLO prediction labels
# Overlapping boxes are counted once (union of rectangles).
# =============================================================================
#
# Plan:
#   1. Parse YOLO label files: each line = class_id x_center y_center width height (normalized 0-1).
#   2. Get image dimensions from the corresponding .jpg (so coords can be converted to pixels).
#   3. Convert each box to pixel coordinates (x1,y1,x2,y2) and then to a polygon.
#   4. Compute the UNION of all rectangles (using sf::st_union); area of that union is the
#      total nodule area for that image, with overlaps counted only once.
#   5. Optionally export per-image and total summary to CSV.
#
# Required: install.packages(c("sf", "magick"))
#   - magick: to read image dimensions without loading full image
#   - sf: to compute union of polygons and area (or we fall back to sum of areas with a warning)
# =============================================================================

# ---- Configuration: set these to your paths ----
# Labels: folder containing .txt YOLO label files (one per image)
LABELS_DIR <- file.path(
  "Indeterminate_Enhanced_Full_yolo11_predictions_maxdet_conf0.15_iou0.6-20260203T030331Z-3-001",
  "Indeterminate_Enhanced_Full_yolo11_predictions_maxdet_conf0.15_iou0.6",
  "labels"
)
# Images: folder containing the corresponding .jpg images (used to get dimensions)
# Use the predictions folder so every label has a matching image:
IMAGES_DIR <- file.path(
  "Indeterminate_Enhanced_Full_yolo11_predictions_maxdet_conf0.15_iou0.6-20260203T030331Z-3-001",
  "Indeterminate_Enhanced_Full_yolo11_predictions_maxdet_conf0.15_iou0.6"
)
# Or use EnhancedData val images (same basenames):
# IMAGES_DIR <- file.path("EnhancedData", "Indeterminate_Enhanced_Model_updated", "data", "images", "val")

# Base directory: project root (where the Indeterminate_Enhanced_Full_yolo... folder lives).
BASE_DIR <- getwd()
# If labels dir not found from getwd(), try directory of this script (when in RStudio)
if (!dir.exists(file.path(BASE_DIR, LABELS_DIR)) && requireNamespace("rstudioapi", quietly = TRUE)) {
  tryCatch({
    ctx <- rstudioapi::getActiveDocumentContext()
    if (!is.null(ctx$path) && nzchar(ctx$path)) {
      script_dir <- dirname(ctx$path)
      if (dir.exists(file.path(script_dir, LABELS_DIR))) {
        BASE_DIR <- script_dir
        message("Using script directory as project root: ", BASE_DIR)
      }
    }
  }, error = function(e) NULL)
}
# Output CSV path (leave NA to skip writing)
OUTPUT_CSV <- "nodule_area_summary.csv"


# ---- Helper: get image dimensions ----
get_image_dimensions <- function(img_path) {
  if (!file.exists(img_path)) return(NULL)
  if (requireNamespace("magick", quietly = TRUE)) {
    img <- magick::image_read(img_path)
    info <- magick::image_info(img)
    return(list(width = info$width, height = info$height))
  }
  if (requireNamespace("jpeg", quietly = TRUE)) {
    d <- dim(jpeg::readJPEG(img_path, native = FALSE))
    return(list(width = d[2L], height = d[1L]))
  }
  stop("Install 'magick' or 'jpeg' to read image dimensions.")
}


# ---- Helper: parse one YOLO label file ----
# Format: class_id x_center y_center width height (normalized 0-1)
read_yolo_labels <- function(path) {
  if (!file.exists(path)) return(data.frame())
  d <- read.table(path, header = FALSE, col.names = c("class", "x_center", "y_center", "width", "height"))
  if (nrow(d) == 0L) return(d)
  d
}


# ---- Helper: convert normalized boxes to pixel coordinates (x1,y1,x2,y2) ----
# YOLO: x_center, y_center are center; width, height are full size (normalized)
norm_to_pixel_boxes <- function(labels, img_width, img_height) {
  if (nrow(labels) == 0L) return(list())
  x1 <- (labels$x_center - labels$width / 2) * img_width
  y1 <- (labels$y_center - labels$height / 2) * img_height
  x2 <- (labels$x_center + labels$width / 2) * img_width
  y2 <- (labels$y_center + labels$height / 2) * img_height
  lapply(seq_len(nrow(labels)), function(i) {
    list(
      x = c(x1[i], x2[i], x2[i], x1[i], x1[i]),
      y = c(y1[i], y1[i], y2[i], y2[i], y1[i])
    )
  })
}


# ---- Helper: area of union of rectangles (overlap counted once) ----
# Uses sf::st_union + st_area for exact union area
union_rectangles_area <- function(boxes_poly) {
  if (length(boxes_poly) == 0L) return(0)
  if (length(boxes_poly) == 1L) {
    x <- boxes_poly[[1]]$x
    y <- boxes_poly[[1]]$y
    return(abs(sum(x[-5] * y[-1] - x[-1] * y[-5])) / 2)  # shoelace
  }
  if (!requireNamespace("sf", quietly = TRUE)) {
    warning("Package 'sf' not installed. Returning sum of box areas (overlaps counted twice).")
    areas <- vapply(boxes_poly, function(p) {
      abs(sum(p$x[-5] * p$y[-1] - p$x[-1] * p$y[-5])) / 2
    }, 0)
    return(sum(areas))
  }
  sfc <- lapply(boxes_poly, function(p) {
    m <- cbind(p$x, p$y)
    sf::st_polygon(list(m))
  })
  combined <- sf::st_union(do.call(c, lapply(sfc, sf::st_sfc)))
  as.numeric(sf::st_area(combined))
}


# ---- Black-pixel method: load image as grayscale (0-1) ----
load_image_grayscale <- function(img_path) {
  if (!file.exists(img_path)) return(NULL)
  if (requireNamespace("jpeg", quietly = TRUE)) {
    img <- jpeg::readJPEG(img_path, native = FALSE)
    # Standard luminance: 0.299*R + 0.587*G + 0.114*B
    if (length(dim(img)) == 3L) {
      gray <- 0.299 * img[, , 1] + 0.587 * img[, , 2] + 0.114 * img[, , 3]
    } else {
      gray <- as.matrix(img)
    }
    return(gray)
  }
  if (requireNamespace("magick", quietly = TRUE)) {
    img <- magick::image_read(img_path)
    img <- magick::image_convert(img, "gray")
    d <- magick::image_data(img, "gray")
    a <- as.numeric(d)
    dim(a) <- dim(d)[1:2]
    a <- t(a)  # magick is (width, height); we need (height, width) to match mask
    if (max(a, na.rm = TRUE) > 1) a <- a / 255
    return(a)
  }
  stop("Install 'jpeg' or 'magick' to load images.")
}


# ---- Build binary mask: 1 inside union of boxes, 0 outside (overlap = 1 once) ----
# boxes_poly: list from norm_to_pixel_boxes (each element has x, y coords)
# Image convention: row = y, col = x; 1-based indices
boxes_to_union_mask <- function(boxes_poly, img_width, img_height) {
  mask <- matrix(0, nrow = img_height, ncol = img_width)
  for (p in boxes_poly) {
    x1 <- max(1, min(p$x)); x2 <- min(img_width, max(p$x))
    y1 <- max(1, min(p$y)); y2 <- min(img_height, max(p$y))
    c1 <- max(1L, floor(x1) + 1L); c2 <- min(img_width, ceiling(x2))
    r1 <- max(1L, floor(y1) + 1L); r2 <- min(img_height, ceiling(y2))
    if (r1 <= r2 && c1 <= c2) mask[r1:r2, c1:c2] <- 1
  }
  mask
}


# ---- Count pixels that are "black" (<= threshold) inside the union mask ----
# gray: matrix (height x width), values 0-1. threshold: 0-1, pixels with gray <= threshold count as black.
count_black_in_mask <- function(gray, mask, threshold) {
  sum((gray <= threshold) & (mask > 0.5), na.rm = TRUE)
}


# ---- Main: process all label files (union-of-boxes area) ----
run_estimation <- function(labels_dir = LABELS_DIR,
                           images_dir = IMAGES_DIR,
                           base_dir = BASE_DIR,
                           output_csv = OUTPUT_CSV) {
  labels_path <- file.path(base_dir, labels_dir)
  images_path <- file.path(base_dir, images_dir)
  if (!dir.exists(labels_path)) {
    stop("Labels directory not found: ", labels_path,
         "\n  Set BASE_DIR to your project root (e.g. the HyperspectralData folder), or run setwd('C:/Users/11579/Desktop/HyperspectralData') before sourcing.")
  }
  if (!dir.exists(images_path)) {
    stop("Images directory not found: ", normalizePath(images_path, mustWork = FALSE))
  }

  label_files <- list.files(labels_path, pattern = "\\.txt$", full.names = FALSE)
  if (length(label_files) == 0L) {
    message("No .txt label files found in ", labels_path)
    return(invisible(NULL))
  }

  results <- data.frame(
    image_id = character(length(label_files)),
    image_file = character(length(label_files)),
    img_width = integer(length(label_files)),
    img_height = integer(length(label_files)),
    n_boxes = integer(length(label_files)),
    nodule_area_px = numeric(length(label_files)),
    nodule_area_frac = numeric(length(label_files)),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(label_files)) {
    fn <- label_files[i]
    base_name <- sub("\\.txt$", "", fn)
    image_file <- paste0(base_name, ".jpg")
    img_path <- file.path(images_path, image_file)
    label_path <- file.path(labels_path, fn)

    dims <- get_image_dimensions(img_path)
    if (is.null(dims)) {
      warning("Image not found, skipping: ", image_file)
      results$image_id[i] <- base_name
      results$image_file[i] <- image_file
      results$n_boxes[i] <- 0L
      results$nodule_area_px[i] <- NA_real_
      results$nodule_area_frac[i] <- NA_real_
      next
    }

    labels <- read_yolo_labels(label_path)
    n_boxes <- nrow(labels)
    results$image_id[i] <- base_name
    results$image_file[i] <- image_file
    results$img_width[i] <- dims$width
    results$img_height[i] <- dims$height
    results$n_boxes[i] <- n_boxes

    if (n_boxes == 0L) {
      results$nodule_area_px[i] <- 0
      results$nodule_area_frac[i] <- 0
      next
    }

    boxes_poly <- norm_to_pixel_boxes(labels, dims$width, dims$height)
    area_px <- union_rectangles_area(boxes_poly)
    img_area <- dims$width * dims$height
    results$nodule_area_px[i] <- area_px
    results$nodule_area_frac[i] <- area_px / img_area
  }

  # Summary
  total_area_px <- sum(results$nodule_area_px, na.rm = TRUE)
  n_images <- sum(!is.na(results$nodule_area_px))
  message("Processed ", n_images, " images.")
  message("Total nodule area (union, overlap counted once): ", round(total_area_px, 2), " px across all images.")
  message("Mean nodule area per image: ", round(total_area_px / max(1, n_images), 2), " px.")

  if (!is.na(output_csv) && nzchar(output_csv)) {
    out_path <- file.path(base_dir, output_csv)
    write.csv(results, out_path, row.names = FALSE)
    message("Wrote: ", out_path)
  }

  invisible(results)
}


# ---- Process all label files: nodule area = black pixels in union of boxes ----
# threshold_grayscale: 0-1; pixels with intensity <= this inside boxes count as nodule. Try 0.25--0.4.
run_estimation_black_pixels <- function(labels_dir,
                                        images_dir,
                                        base_dir,
                                        threshold_grayscale = 0.35,
                                        output_csv = NA) {
  labels_path <- file.path(base_dir, labels_dir)
  images_path <- file.path(base_dir, images_dir)
  if (!dir.exists(labels_path)) stop("Labels directory not found: ", labels_path)
  if (!dir.exists(images_path)) stop("Images directory not found: ", images_path)

  label_files <- list.files(labels_path, pattern = "\\.txt$", full.names = FALSE)
  if (length(label_files) == 0L) {
    message("No .txt label files found in ", labels_path)
    return(invisible(NULL))
  }

  results <- data.frame(
    image_id = character(length(label_files)),
    image_file = character(length(label_files)),
    img_width = integer(length(label_files)),
    img_height = integer(length(label_files)),
    n_boxes = integer(length(label_files)),
    nodule_area_px = numeric(length(label_files)),
    nodule_area_frac = numeric(length(label_files)),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(label_files)) {
    fn <- label_files[i]
    base_name <- sub("\\.txt$", "", fn)
    image_file <- paste0(base_name, ".jpg")
    img_path <- file.path(images_path, image_file)
    label_path <- file.path(labels_path, fn)

    labels <- read_yolo_labels(label_path)
    n_boxes <- nrow(labels)
    results$image_id[i] <- base_name
    results$image_file[i] <- image_file
    results$n_boxes[i] <- n_boxes

    if (!file.exists(img_path)) {
      warning("Image not found, skipping: ", image_file)
      results$img_width[i] <- NA_integer_
      results$img_height[i] <- NA_integer_
      results$nodule_area_px[i] <- NA_real_
      results$nodule_area_frac[i] <- NA_real_
      next
    }

    gray <- load_image_grayscale(img_path)
    if (is.null(gray)) {
      results$img_width[i] <- NA_integer_
      results$img_height[i] <- NA_integer_
      results$nodule_area_px[i] <- NA_real_
      results$nodule_area_frac[i] <- NA_real_
      next
    }
    img_height <- nrow(gray)
    img_width <- ncol(gray)
    results$img_width[i] <- img_width
    results$img_height[i] <- img_height
    img_area <- img_width * img_height

    if (n_boxes == 0L) {
      results$nodule_area_px[i] <- 0
      results$nodule_area_frac[i] <- 0
      next
    }

    boxes_poly <- norm_to_pixel_boxes(labels, img_width, img_height)
    mask <- boxes_to_union_mask(boxes_poly, img_width, img_height)
    black_count <- count_black_in_mask(gray, mask, threshold_grayscale)
    results$nodule_area_px[i] <- black_count
    results$nodule_area_frac[i] <- black_count / img_area
  }

  n_images <- sum(!is.na(results$nodule_area_px))
  total_black <- sum(results$nodule_area_px, na.rm = TRUE)
  message("Processed ", n_images, " images (black-pixel method, threshold = ", threshold_grayscale, ").")
  message("Total nodule (black) pixels: ", round(total_black, 0), ".")

  if (!is.na(output_csv) && nzchar(output_csv)) {
    write.csv(results, file.path(base_dir, output_csv), row.names = FALSE)
    message("Wrote: ", file.path(base_dir, output_csv))
  }
  invisible(results)
}


# ---- Run (skip when sourced by nodule_area_datasets_yolo11.R) ----
if (!exists("DONT_RUN_ESTIMATION") || !isTRUE(DONT_RUN_ESTIMATION)) {
  if (!dir.exists(file.path(BASE_DIR, LABELS_DIR)) && requireNamespace("rstudioapi", quietly = TRUE)) {
    tryCatch({
      ctx <- rstudioapi::getActiveDocumentContext()
      if (!is.null(ctx$path) && nzchar(ctx$path)) {
        script_dir <- dirname(ctx$path)
        if (dir.exists(file.path(script_dir, LABELS_DIR))) BASE_DIR <- script_dir
      }
    }, error = function(e) NULL)
  }
  setwd(BASE_DIR)
  results <- run_estimation(
    labels_dir = LABELS_DIR,
    images_dir = IMAGES_DIR,
    base_dir = BASE_DIR,
    output_csv = OUTPUT_CSV
  )
  if (!is.null(results)) print(head(results, 10))
}
