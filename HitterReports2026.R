#===============================================================================
# HITTer REPORTS SHINY APP (WITH PDF EXPORT)
#   - Added PDF generation for all hitters from most recent date
#   - Added PA-by-PA strike zone plots (Shiny + PDF)
#   - Original Shiny app functionality preserved
#===============================================================================

library(shiny)
library(ggplot2)
library(dplyr)
library(stringr)
library(showtext)
library(gt)
library(lightgbm)
library(xgboost)
library(png)
library(grid)
library(gridExtra)

# -------------------- Font Setup --------------------
library(sysfonts)
tmp <- file.path(getwd(), "tmp_showtext")
if (!dir.exists(tmp)) dir.create(tmp, recursive = TRUE)
Sys.setenv(TMPDIR = tmp)

ok <- TRUE
tryCatch(
  sysfonts::font_add_google("Markazi Text", "markazi", regular.wt = 600),
  error = function(e) { message("Font fetch failed"); ok <<- FALSE }
)
if (!ok) sysfonts::font_add(family = "markazi", regular = "")
showtext_auto()

# -------------------- Load Models from Disk --------------------
models_path <- "/Users/a13105/Documents/R Projects/Scripts/Pitcher Reports 2026/www/"

cat("Loading models from disk...\n")

# LIHAA/LIVAA LINEAR MODELS
VAA_model_trained <- tryCatch({
  model <- readRDS(paste0(models_path, "VAA_model.rds"))
  cat("  - VAA_model loaded successfully\n")
  model
}, error = function(e) {
  warning("Could not load VAA_model: ", e$message)
  NULL
})

HAA_model_trained <- tryCatch({
  model <- readRDS(paste0(models_path, "HAA_model.rds"))
  cat("  - HAA_model loaded successfully\n")
  model
}, error = function(e) {
  warning("Could not load HAA_model: ", e$message)
  NULL
})

# OPTION B CALLED-PITCH MODELS
xgb_model_hbp <- tryCatch({
  model <- xgb.load(paste0(models_path, "xgb_model_hbp.model"))
  cat("  - xgb_model_hbp loaded successfully\n")
  model
}, error = function(e) {
  warning("Could not load xgb_model_hbp: ", e$message)
  NULL
})

xgb_model_ballstrike <- tryCatch({
  model <- xgb.load(paste0(models_path, "xgb_model_ballstrike.model"))
  cat("  - xgb_model_ballstrike loaded successfully\n")
  model
}, error = function(e) {
  warning("Could not load xgb_model_ballstrike: ", e$message)
  NULL
})

# LightGBM models
final_model_xSwing <- tryCatch({
  lgb.load(paste0(models_path, "final_model_xSwing.txt"))
}, error = function(e) {
  warning("Could not load final_model_xSwing: ", e$message)
  NULL
})

final_model_swings <- tryCatch({
  lgb.load(paste0(models_path, "final_model_swings.txt"))
}, error = function(e) {
  warning("Could not load final_model_swings: ", e$message)
  NULL
})

final_model_contact <- tryCatch({
  lgb.load(paste0(models_path, "final_model_contact.txt"))
}, error = function(e) {
  warning("Could not load final_model_contact: ", e$message)
  NULL
})

models_in_play <- list(
  ground_ball = tryCatch({ lgb.load(paste0(models_path, "final_model_in_play_ground_ball.txt")) },
                         error = function(e) { warning("Could not load ground_ball model: ", e$message); NULL }),
  line_drive  = tryCatch({ lgb.load(paste0(models_path, "final_model_in_play_line_drive.txt"))  },
                         error = function(e) { warning("Could not load line_drive model: ", e$message); NULL }),
  fly_ball    = tryCatch({ lgb.load(paste0(models_path, "final_model_in_play_fly_ball.txt"))    },
                         error = function(e) { warning("Could not load fly_ball model: ", e$message); NULL }),
  pop_up      = tryCatch({ lgb.load(paste0(models_path, "final_model_in_play_pop_up.txt"))      },
                         error = function(e) { warning("Could not load pop_up model: ", e$message); NULL })
)

cat("Models loaded from disk:\n")
cat("  - VAA_model:", !is.null(VAA_model_trained), "\n")
cat("  - HAA_model:", !is.null(HAA_model_trained), "\n")
cat("  - xgb_model_hbp:", !is.null(xgb_model_hbp), "\n")
cat("  - xgb_model_ballstrike:", !is.null(xgb_model_ballstrike), "\n")
cat("  - final_model_xSwing:", !is.null(final_model_xSwing), "\n")
cat("  - final_model_swings:", !is.null(final_model_swings), "\n")
cat("  - final_model_contact:", !is.null(final_model_contact), "\n")
cat("  - in_play models:", sum(!sapply(models_in_play, is.null)), "of 4\n")

# ==================== RELATIVE RELEASE CONTROLS ====================
ANCHOR_TRACKMAN_SIDE   <- 1.2
ANCHOR_TRACKMAN_HEIGHT <- 5.8
ANCHOR_PLOT_X <- -0.4
ANCHOR_PLOT_Z <- 4.85

RUBBER_TRACKMAN_SIDE   <- 0
RUBBER_TRACKMAN_HEIGHT <- 0
RUBBER_PLOT_X <- 0
RUBBER_PLOT_Z <- 3.5

SCALE_X <- (ANCHOR_PLOT_X - RUBBER_PLOT_X) / (ANCHOR_TRACKMAN_SIDE - RUBBER_TRACKMAN_SIDE)
SCALE_Z <- (ANCHOR_PLOT_Z - RUBBER_PLOT_Z) / (ANCHOR_TRACKMAN_HEIGHT - RUBBER_TRACKMAN_HEIGHT)

trackman_to_plot <- function(tm_side, tm_height) {
  plot_x <- RUBBER_PLOT_X + (tm_side - RUBBER_TRACKMAN_SIDE) * SCALE_X
  plot_z <- RUBBER_PLOT_Z + (tm_height - RUBBER_TRACKMAN_HEIGHT) * SCALE_Z
  list(x = plot_x, z = plot_z)
}

# -------------------- Run Values --------------------
run_values <- c(
  "0-0" = 0.001695997, "1-0" = 0.039248042, "0-1" = -0.043581338,
  "2-0" = 0.077612355, "1-1" = -0.015277684, "0-2" = -0.103242476,
  "3-0" = 0.200960731, "2-1" = 0.034545018, "1-2" = -0.080485991,
  "3-1" = 0.138254876, "2-2" = -0.039716495, "3-2" = 0.048505049,
  "Walk" = 0.325, "Strikeout" = -0.284, "HitByPitch" = 0.325,
  "FlyBall" = 0.586, "LineDrive" = 0.528,
  "GroundBall" = 0.164, "PopUp" = 0.0186
)

# -------------------- Pitch Colors --------------------
pitch_colors <- c(
  "Fastball" = "#D22D49", "Sinker" = "#FE9D00", "Cutter" = "#933F2C",
  "Slider" = "#EEE716", "Curveball" = "#00D1ED", "Splitter" = "#3BACAC",
  "ChangeUp" = "#1DBE3A", "Sweeper" = "#DDB33A"
)

# -------------------- Training Pitch Types --------------------
TRAINING_PITCH_TYPES_VAA_HAA <- c(
  "ChangeUp", "Curveball", "Cutter", "Fastball",
  "Knuckleball", "Other", "Sinker", "Slider",
  "Splitter", "Sweeper"
)

# -------------------- Helper Functions --------------------
`%||%` <- function(a, b) if (!is.null(a)) a else b

mirror_plate_side <- function(x) {
  xv <- suppressWarnings(as.numeric(x))
  ifelse(is.na(xv), NA_real_, -xv)
}

called_features <- c(
  "PlateLocSide", "PlateLocHeight",
  "BatterSide", "PitcherThrows",
  "RelSpeed", "ax0", "az0",
  "RelSide", "RelHeight", "Extension"
)

get_xgb_feature_names <- function(model, fallback = NULL) {
  if (is.null(model)) return(fallback)
  fn <- tryCatch(model$feature_names, error = function(e) NULL)
  if (is.null(fn) || length(fn) == 0) return(fallback)
  fn
}

split_if_single_string <- function(fn) {
  if (is.null(fn)) return(NULL)
  fn <- as.character(fn)
  
  if (any(grepl("\\s", fn))) {
    fn <- unlist(strsplit(fn, "\\s+"))
    fn <- fn[nzchar(fn)]
  }
  
  fn
}

extract_lgb_feature_names_from_txt <- function(path) {
  if (is.null(path) || !file.exists(path)) return(NULL)
  x <- readLines(path, warn = FALSE)
  ln <- x[grepl("^feature_names=", x)]
  if (length(ln) == 0) return(NULL)
  raw <- sub("^feature_names=", "", ln[1])
  fn <- unlist(strsplit(raw, "\\s+"))
  fn <- fn[nzchar(fn)]
  if (length(fn) == 0) NULL else fn
}

get_lgb_feature_names <- function(model, txt_path = NULL, fallback = NULL) {
  if (is.null(model)) return(split_if_single_string(fallback))
  
  fn <- tryCatch(lgb.get.feat.name(model), error = function(e) NULL)
  fn <- split_if_single_string(fn)
  
  if (is.null(fn) || length(fn) == 0) {
    fn <- extract_lgb_feature_names_from_txt(txt_path)
    fn <- split_if_single_string(fn)
  }
  
  if ((is.null(fn) || length(fn) == 0) && !is.null(fallback)) {
    fn <- split_if_single_string(fallback)
  }
  
  fn
}

EXPECTED_LGBM_FEATURES <- NULL
if (!is.null(final_model_xSwing)) {
  EXPECTED_LGBM_FEATURES <- get_lgb_feature_names(
    final_model_xSwing,
    txt_path = paste0(models_path, "final_model_xSwing.txt")
  )
}

EXPECTED_LGBM_FEATURES_SWINGS  <- NULL
EXPECTED_LGBM_FEATURES_CONTACT <- NULL
EXPECTED_LGBM_FEATURES_BB      <- list()

if (!is.null(final_model_swings)) {
  EXPECTED_LGBM_FEATURES_SWINGS <- get_lgb_feature_names(
    final_model_swings,
    txt_path = paste0(models_path, "final_model_swings.txt")
  )
}

if (!is.null(final_model_contact)) {
  EXPECTED_LGBM_FEATURES_CONTACT <- get_lgb_feature_names(
    final_model_contact,
    txt_path = paste0(models_path, "final_model_contact.txt")
  )
}

for (nm in names(models_in_play)) {
  m <- models_in_play[[nm]]
  if (!is.null(m)) {
    EXPECTED_LGBM_FEATURES_BB[[nm]] <- get_lgb_feature_names(
      m,
      txt_path = paste0(models_path, "final_model_in_play_", nm, ".txt")
    )
  } else {
    EXPECTED_LGBM_FEATURES_BB[[nm]] <- NULL
  }
}

standardize_pitch_type <- function(x) {
  x <- as.character(x)
  dplyr::case_when(
    x %in% c("Changeup", "ChangeUp") ~ "ChangeUp",
    x %in% c("Fastball", "FourSeamFastBall") ~ "Fastball",
    x %in% c("Sinker", "TwoSeamFastBall") ~ "Sinker",
    x == "OneSeamFastBall" ~ "Fastball",
    x %in% c("Undefined", "Other") ~ "Other",
    TRUE ~ x
  )
}

lgb_align_matrix <- function(model, df, txt_path = NULL, fallback_feature_names = NULL) {
  fn <- get_lgb_feature_names(model, txt_path = txt_path, fallback = fallback_feature_names)
  if (is.null(fn) || length(fn) == 0) stop("No feature list available for LightGBM alignment.")
  
  for (f in fn) if (!f %in% names(df)) df[[f]] <- 0
  X <- as.matrix(df[, fn, drop = FALSE])
  colnames(X) <- fn
  X
}

CONTACT_MODEL_TARGET <- "given_contact"

compute_outcome_percentages <- function(xSwing, xWhiff, xIn_Play, xGB, xLD, xFB, xPU,
                                        xBall, xStrike, xHBP,
                                        target = CONTACT_MODEL_TARGET) {
  clamp01 <- function(v) pmin(pmax(as.numeric(v), 0), 1)
  
  xSwing   <- clamp01(xSwing)
  xWhiff   <- clamp01(xWhiff)
  xIn_Play <- clamp01(xIn_Play)
  
  xBall   <- clamp01(xBall)
  xStrike <- clamp01(xStrike)
  xHBP    <- clamp01(xHBP)
  
  xGB <- clamp01(xGB); xLD <- clamp01(xLD); xFB <- clamp01(xFB); xPU <- clamp01(xPU)
  
  bb_sum <- xGB + xLD + xFB + xPU
  bb_sum[bb_sum == 0] <- 1
  xGB <- xGB / bb_sum
  xLD <- xLD / bb_sum
  xFB <- xFB / bb_sum
  xPU <- xPU / bb_sum
  
  if (identical(target, "given_contact")) {
    p_whiff  <- xSwing * xWhiff
    p_inplay <- xSwing * (1 - xWhiff) * xIn_Play
    p_foul   <- xSwing * (1 - xWhiff) * (1 - xIn_Play)
  } else if (identical(target, "among_swings")) {
    raw_whiff  <- xWhiff
    raw_inplay <- xIn_Play
    raw_foul   <- pmax(0, 1 - raw_whiff - raw_inplay)
    
    s <- raw_whiff + raw_inplay + raw_foul
    s[s == 0] <- 1
    raw_whiff  <- raw_whiff / s
    raw_inplay <- raw_inplay / s
    raw_foul   <- raw_foul / s
    
    p_whiff  <- xSwing * raw_whiff
    p_inplay <- xSwing * raw_inplay
    p_foul   <- xSwing * raw_foul
  } else {
    stop("CONTACT_MODEL_TARGET must be 'given_contact' or 'among_swings'")
  }
  
  call_sum <- xBall + xStrike + xHBP
  call_sum[call_sum == 0] <- 1
  xBall   <- xBall / call_sum
  xStrike <- xStrike / call_sum
  xHBP    <- xHBP / call_sum
  
  p_ball    <- (1 - xSwing) * xBall
  p_cstrike <- (1 - xSwing) * xStrike
  p_hbp     <- (1 - xSwing) * xHBP
  
  p_gb <- p_inplay * xGB
  p_ld <- p_inplay * xLD
  p_fb <- p_inplay * xFB
  p_pu <- p_inplay * xPU
  
  data.frame(
    pct_Whiff = 100 * p_whiff,
    pct_Contact_Foul = 100 * p_foul,
    pct_Contact_InPlay_GB = 100 * p_gb,
    pct_Contact_InPlay_LD = 100 * p_ld,
    pct_Contact_InPlay_FB = 100 * p_fb,
    pct_Contact_InPlay_PU = 100 * p_pu,
    pct_CalledStrike = 100 * p_cstrike,
    pct_Ball = 100 * p_ball,
    pct_HBP = 100 * p_hbp
  )
}

add_LI_features <- function(df) {
  if (all(c("LIHAA", "LIVAA") %in% names(df))) return(df)
  
  needed <- c("TaggedPitchType", "PlateLocSide", "PlateLocHeight", "VertApprAngle", "HorzApprAngle")
  miss <- setdiff(needed, names(df))
  if (length(miss) > 0) {
    warning("Cannot compute LIHAA/LIVAA; missing columns: ", paste(miss, collapse = ", "))
    df$LIHAA <- NA_real_
    df$LIVAA <- NA_real_
    return(df)
  }
  
  if (is.null(VAA_model_trained) || is.null(HAA_model_trained)) {
    stop("CRITICAL: VAA_model and HAA_model must be loaded!")
  }
  
  df <- df %>%
    mutate(TaggedPitchType_LI = case_when(
      TaggedPitchType %in% c("Changeup", "ChangeUp") ~ "ChangeUp",
      TaggedPitchType %in% c("Fastball", "FourSeamFastBall") ~ "Fastball",
      TaggedPitchType %in% c("Sinker", "TwoSeamFastBall") ~ "Sinker",
      TaggedPitchType == "OneSeamFastBall" ~ "Fastball",
      TaggedPitchType %in% c("Undefined", "Other") ~ "Other",
      TRUE ~ as.character(TaggedPitchType)
    ))
  
  df$TaggedPitchType_LI <- factor(df$TaggedPitchType_LI, levels = TRAINING_PITCH_TYPES_VAA_HAA)
  
  vaa_onehot <- model.matrix(~ TaggedPitchType_LI - 1, data = df)
  colnames(vaa_onehot) <- sub("^TaggedPitchType_LI", "TaggedPitchType", colnames(vaa_onehot))
  
  vaa_data <- data.frame(
    VertApprAngle = df$VertApprAngle,
    PlateLocSide  = df$PlateLocSide
  )
  vaa_data <- cbind(vaa_data, vaa_onehot)
  
  expected_vaa_cols <- c("PlateLocSide", paste0("TaggedPitchType", TRAINING_PITCH_TYPES_VAA_HAA))
  for (col in expected_vaa_cols) if (!col %in% names(vaa_data)) vaa_data[[col]] <- 0
  vaa_data <- vaa_data[, c("VertApprAngle", expected_vaa_cols), drop = FALSE]
  
  df$xVAA <- tryCatch({
    as.numeric(predict(VAA_model_trained, newdata = vaa_data))
  }, error = function(e) {
    warning("VAA prediction failed: ", e$message)
    df$VertApprAngle
  })
  
  haa_onehot <- model.matrix(~ TaggedPitchType_LI - 1, data = df)
  colnames(haa_onehot) <- sub("^TaggedPitchType_LI", "TaggedPitchType", colnames(haa_onehot))
  
  haa_data <- data.frame(
    HorzApprAngle  = df$HorzApprAngle,
    PlateLocHeight = df$PlateLocHeight
  )
  haa_data <- cbind(haa_data, haa_onehot)
  
  expected_haa_cols <- c("PlateLocHeight", paste0("TaggedPitchType", TRAINING_PITCH_TYPES_VAA_HAA))
  for (col in expected_haa_cols) if (!col %in% names(haa_data)) haa_data[[col]] <- 0
  haa_data <- haa_data[, c("HorzApprAngle", expected_haa_cols), drop = FALSE]
  
  df$xHAA <- tryCatch({
    as.numeric(predict(HAA_model_trained, newdata = haa_data))
  }, error = function(e) {
    warning("HAA prediction failed: ", e$message)
    df$HorzApprAngle
  })
  
  df <- df %>%
    mutate(
      LIVAA = xVAA - VertApprAngle,
      LIHAA = xHAA - HorzApprAngle
    ) %>%
    select(-TaggedPitchType_LI)
  
  df
}

pitch_equation <- function(a, v, p0, t) p0 + v * t + 0.5 * a * t^2

ellipse_df <- function(cx, cy, width, height, n = 300) {
  th <- seq(0, 2*pi, length.out = n)
  data.frame(
    x = cx + (width/2)  * cos(th),
    y = cy + (height/2) * sin(th)
  )
}

safe_read_png <- function(path) {
  if (is.null(path) || is.na(path) || path == "" || !file.exists(path)) return(NULL)
  tryCatch(png::readPNG(path), error = function(e) NULL)
}

mound_elev_at_release <- function(extension_ft, mound_height_ft = 10/12, rubber_to_plate_ft = 60.5) {
  ext <- suppressWarnings(as.numeric(extension_ft))
  if (!is.finite(ext)) ext <- 6
  release_dist <- rubber_to_plate_ft - ext
  release_dist <- max(0, min(rubber_to_plate_ft, release_dist))
  mound_height_ft * (release_dist / rubber_to_plate_ft)
}

clamp01 <- function(v) pmin(pmax(as.numeric(v), 0), 1)

normalize3 <- function(a, b, c) {
  s <- a + b + c
  s[s == 0] <- 1
  list(a = a / s, b = b / s, c = c / s)
}

normalize4 <- function(a, b, c, d) {
  s <- a + b + c + d
  s[s == 0] <- 1
  list(a = a / s, b = b / s, c = c / s, d = d / s)
}

calculate_expected_outcomes <- function(df) {
  
  if (nrow(df) == 0) return(df)
  df <- as.data.frame(df)
  
  if (!"row_index" %in% names(df)) df$row_index <- seq_len(nrow(df))
  
  df <- add_LI_features(df)
  
  df_swing <- df %>%
    mutate(
      BatterSide    = ifelse(BatterSide == "Right", 1, 0),
      PitcherThrows = ifelse(PitcherThrows == "Right", 1, 0),
      TaggedPitchType = standardize_pitch_type(TaggedPitchType)
    )
  
  training_pitch_types <- c(
    "ChangeUp","Curveball","Cutter",
    "Fastball","FourSeamFastBall",
    "Knuckleball","Other","Undefined",
    "Sinker","TwoSeamFastBall",
    "Slider","Splitter","Sweeper"
  )
  
  df_tmp <- df_swing
  df_tmp$TaggedPitchType <- factor(df_tmp$TaggedPitchType, levels = training_pitch_types)
  TaggedPitchType_oh <- model.matrix(~ TaggedPitchType - 1, df_tmp)
  
  df_xswing_feat <- cbind(
    df_tmp %>% transmute(
      PitcherThrows, Balls, Strikes, BatterSide,
      SpinRate, SpinAxis, RelSpeed, az0, ax0,
      RelSide, RelHeight, Extension,
      PlateLocSide, PlateLocHeight,
      row_index, LIHAA, LIVAA
    ),
    TaggedPitchType_oh
  )
  
  df_contact_feat <- df_swing %>%
    transmute(
      Balls, Strikes,
      PitcherThrows, BatterSide,
      SpinRate, SpinAxis,
      RelSpeed, az0, ax0,
      RelSide, RelHeight, Extension,
      PlateLocSide, PlateLocHeight,
      LIHAA, LIVAA
    )
  
  df_xswing_feat[is.na(df_xswing_feat)] <- 0
  df_contact_feat[is.na(df_contact_feat)] <- 0
  
  clamp01 <- function(x) pmin(pmax(as.numeric(x), 0), 1)
  
  df$xSwing <- rep(0.5, nrow(df))
  if (!is.null(final_model_xSwing)) {
    X <- lgb_align_matrix(
      final_model_xSwing,
      df_xswing_feat,
      txt_path = paste0(models_path, "final_model_xSwing.txt"),
      fallback_feature_names = EXPECTED_LGBM_FEATURES
    )
    if (nrow(X) == nrow(df) && nrow(X) > 0) {
      p <- tryCatch(predict(final_model_xSwing, X), error = function(e) NULL)
      if (!is.null(p) && length(p) == nrow(df)) df$xSwing <- p
    }
  }
  
  df$xWhiff <- rep(0.3, nrow(df))
  if (!is.null(final_model_swings)) {
    X <- lgb_align_matrix(
      final_model_swings,
      df_xswing_feat,
      txt_path = paste0(models_path, "final_model_swings.txt"),
      fallback_feature_names = EXPECTED_LGBM_FEATURES_SWINGS
    )
    if (nrow(X) == nrow(df) && nrow(X) > 0) {
      p <- tryCatch(predict(final_model_swings, X), error = function(e) NULL)
      if (!is.null(p) && length(p) == nrow(df)) df$xWhiff <- p
    }
  }
  
  df$xIn_Play <- rep(0.5, nrow(df))
  if (!is.null(final_model_contact)) {
    Xc <- lgb_align_matrix(
      final_model_contact,
      df_contact_feat,
      txt_path = paste0(models_path, "final_model_contact.txt"),
      fallback_feature_names = EXPECTED_LGBM_FEATURES_CONTACT
    )
    if (nrow(Xc) == nrow(df) && nrow(Xc) > 0) {
      p <- tryCatch(predict(final_model_contact, Xc), error = function(e) NULL)
      if (!is.null(p) && length(p) == nrow(df)) df$xIn_Play <- p
    }
  }
  
  bb_map <- c(ground_ball="xGB", line_drive="xLD", fly_ball="xFB", pop_up="xPU")
  for (nm in names(bb_map)) {
    out <- bb_map[[nm]]
    df[[out]] <- rep(0.25, nrow(df))
    m <- models_in_play[[nm]]
    if (!is.null(m)) {
      Xb <- lgb_align_matrix(
        m,
        df_xswing_feat,
        txt_path = paste0(models_path, "final_model_in_play_", nm, ".txt"),
        fallback_feature_names = EXPECTED_LGBM_FEATURES_BB[[nm]]
      )
      if (nrow(Xb) == nrow(df) && nrow(Xb) > 0) {
        p <- tryCatch(predict(m, Xb), error = function(e) NULL)
        if (!is.null(p) && length(p) == nrow(df)) df[[out]] <- p
      }
    }
  }
  
  df_called <- df_swing %>%
    select(all_of(intersect(called_features, names(df_swing))))
  
  for (feat in called_features) {
    if (!feat %in% names(df_called)) df_called[[feat]] <- 0
  }
  
  df_called[is.na(df_called)] <- 0
  
  if (!is.null(xgb_model_hbp)) {
    feat_names_hbp <- get_xgb_feature_names(xgb_model_hbp, fallback = called_features)
    for (f in feat_names_hbp) if (!f %in% names(df_called)) df_called[[f]] <- 0
    
    X_hbp <- as.matrix(df_called[, feat_names_hbp, drop = FALSE])
    
    pred_hbp <- tryCatch({
      predict(xgb_model_hbp, X_hbp)
    }, error = function(e) {
      warning("xHBP prediction failed: ", e$message)
      rep(0.01, nrow(df))
    })
    df$xHBP <- clamp01(pred_hbp)
  } else {
    df$xHBP <- rep(0.01, nrow(df))
  }
  
  if (!is.null(xgb_model_ballstrike)) {
    feat_names_bs <- get_xgb_feature_names(xgb_model_ballstrike, fallback = called_features)
    for (f in feat_names_bs) if (!f %in% names(df_called)) df_called[[f]] <- 0
    
    X_bs <- as.matrix(df_called[, feat_names_bs, drop = FALSE])
    
    pred_strike <- tryCatch({
      predict(xgb_model_ballstrike, X_bs)
    }, error = function(e) {
      warning("xBallStrike prediction failed: ", e$message)
      rep(0.5, nrow(df))
    })
    
    df$xStrike <- clamp01(pred_strike)
    df$xBall <- 1 - df$xStrike
  } else {
    df$xBall <- rep(0.5, nrow(df))
    df$xStrike <- rep(0.5, nrow(df))
  }
  
  df$xSwing   <- clamp01(df$xSwing)
  df$xWhiff   <- clamp01(df$xWhiff)
  df$xIn_Play <- clamp01(df$xIn_Play)
  
  df$xBall   <- clamp01(df$xBall)
  df$xStrike <- clamp01(df$xStrike)
  df$xHBP    <- clamp01(df$xHBP)
  
  df$xGB <- clamp01(df$xGB); df$xLD <- clamp01(df$xLD)
  df$xFB <- clamp01(df$xFB); df$xPU <- clamp01(df$xPU)
  
  call_sum <- df$xBall + df$xStrike + df$xHBP
  call_sum[call_sum == 0] <- 1
  df$xBall   <- df$xBall / call_sum
  df$xStrike <- df$xStrike / call_sum
  df$xHBP    <- df$xHBP / call_sum
  
  bb_sum <- df$xGB + df$xLD + df$xFB + df$xPU
  bb_sum[bb_sum == 0] <- 1
  df$xGB <- df$xGB / bb_sum
  df$xLD <- df$xLD / bb_sum
  df$xFB <- df$xFB / bb_sum
  df$xPU <- df$xPU / bb_sum
  
  swing_mass  <- df$xSwing
  called_mass <- 1 - swing_mass
  
  xContact <- 1 - df$xWhiff
  xFoul    <- 1 - df$xIn_Play
  
  df$pct_Whiff             <- 100 * swing_mass * df$xWhiff
  df$pct_Contact_Foul      <- 100 * swing_mass * xContact * xFoul
  df$pct_Contact_InPlay_GB <- 100 * swing_mass * xContact * df$xIn_Play * df$xGB
  df$pct_Contact_InPlay_LD <- 100 * swing_mass * xContact * df$xIn_Play * df$xLD
  df$pct_Contact_InPlay_FB <- 100 * swing_mass * xContact * df$xIn_Play * df$xFB
  df$pct_Contact_InPlay_PU <- 100 * swing_mass * xContact * df$xIn_Play * df$xPU
  
  df$pct_CalledStrike <- 100 * called_mass * df$xStrike
  df$pct_Ball         <- 100 * called_mass * df$xBall
  df$pct_HBP          <- 100 * called_mass * df$xHBP
  
  df$pct_Contact_InPlay_Total <- df$pct_Contact_InPlay_GB +
    df$pct_Contact_InPlay_LD +
    df$pct_Contact_InPlay_FB +
    df$pct_Contact_InPlay_PU
  
  eps <- (df$row_index %% 1000) * 1e-12
  
  inplay <- df$pct_Contact_InPlay_Total + eps*1
  cs <- df$pct_CalledStrike      + eps*2
  bl <- df$pct_Ball              + eps*3
  hb <- df$pct_HBP               + eps*4
  wh <- df$pct_Whiff             + eps*5
  fo <- df$pct_Contact_Foul      + eps*6
  
  mx <- pmax(inplay, cs, bl, hb, wh, fo)
  
  df$outcome_category <- dplyr::case_when(
    inplay == mx ~ "InPlay",
    cs == mx ~ "CalledStrike",
    bl == mx ~ "CalledBall",
    hb == mx ~ "HitByPitch",
    wh == mx ~ "Whiff",
    fo == mx ~ "Foul",
    TRUE ~ NA_character_
  )
  
  df$expected_event <- dplyr::case_when(
    df$outcome_category != "InPlay" ~ df$outcome_category,
    df$outcome_category == "InPlay" ~ {
      gb <- df$pct_Contact_InPlay_GB + eps*1
      ld <- df$pct_Contact_InPlay_LD + eps*2
      fb <- df$pct_Contact_InPlay_FB + eps*3
      pu <- df$pct_Contact_InPlay_PU + eps*4
      
      mx_bb <- pmax(gb, ld, fb, pu)
      
      dplyr::case_when(
        gb == mx_bb ~ "GroundBall",
        ld == mx_bb ~ "LineDrive",
        fb == mx_bb ~ "FlyBall",
        pu == mx_bb ~ "PopUp",
        TRUE ~ "GroundBall"
      )
    },
    TRUE ~ NA_character_
  )
  
  df <- df %>% select(-outcome_category, -pct_Contact_InPlay_Total)
  
  df
}

calculate_swing_decision_rv <- function(df) {
  
  if (!"expected_event" %in% names(df)) {
    df <- calculate_expected_outcomes(df)
  }
  
  if (!"actual_event" %in% names(df)) {
    if ("PitchCall" %in% names(df)) {
      if ("Angle" %in% names(df)) {
        df <- df %>%
          mutate(
            batted_ball_type = case_when(
              PitchCall != "InPlay" ~ NA_character_,
              Angle < 10 ~ "GroundBall",
              Angle >= 10 & Angle < 25 ~ "LineDrive",
              Angle >= 25 & Angle < 50 ~ "FlyBall",
              Angle >= 50 ~ "PopUp"
            )
          )
      } else {
        df$batted_ball_type <- NA_character_
      }
      
      df <- df %>%
        mutate(
          actual_event = case_when(
            PitchCall %in% c("BallCalled", "BallinDirt", "BallInDirt") ~ "CalledBall",
            PitchCall == "StrikeCalled" ~ "CalledStrike",
            PitchCall %in% c("FoulBall", "FoulBallFieldable", "FoulBallNotFieldable") ~ "Foul",
            PitchCall == "StrikeSwinging" ~ "Whiff",
            PitchCall == "HitByPitch" ~ "HitByPitch",
            batted_ball_type == "GroundBall" ~ "GroundBall",
            batted_ball_type == "LineDrive" ~ "LineDrive",
            batted_ball_type == "FlyBall" ~ "FlyBall",
            batted_ball_type == "PopUp" ~ "PopUp",
            TRUE ~ NA_character_
          )
        )
    } else {
      df$actual_event <- NA_character_
    }
  }
  
  df <- df %>%
    mutate(
      next_count_strike = case_when(
        Strikes < 2 ~ paste0(Balls, "-", Strikes + 1),
        Strikes == 2 ~ "Strikeout",
        TRUE ~ paste0(Balls, "-", Strikes)
      ),
      next_count_ball = case_when(
        Balls < 3 ~ paste0(Balls + 1, "-", Strikes),
        Balls == 3 ~ "Walk",
        TRUE ~ paste0(Balls, "-", Strikes)
      ),
      
      expected_rv = case_when(
        expected_event == "CalledBall" ~ run_values[next_count_ball],
        expected_event == "CalledStrike" ~ run_values[next_count_strike],
        expected_event == "Whiff" ~ run_values[next_count_strike],
        expected_event == "Foul" ~ case_when(
          Strikes < 2 ~ run_values[paste0(Balls, "-", Strikes + 1)],
          Strikes == 2 ~ run_values[paste0(Balls, "-", 2)],
          TRUE ~ 0
        ),
        expected_event == "GroundBall" ~ run_values["GroundBall"],
        expected_event == "LineDrive" ~ run_values["LineDrive"],
        expected_event == "FlyBall" ~ run_values["FlyBall"],
        expected_event == "PopUp" ~ run_values["PopUp"],
        expected_event == "HitByPitch" ~ run_values["HitByPitch"],
        TRUE ~ 0
      ),
      
      actual_rv = case_when(
        actual_event == "CalledBall" ~ run_values[next_count_ball],
        actual_event == "CalledStrike" ~ run_values[next_count_strike],
        actual_event == "Whiff" ~ run_values[next_count_strike],
        actual_event == "Foul" ~ case_when(
          Strikes < 2 ~ run_values[paste0(Balls, "-", Strikes + 1)],
          Strikes == 2 ~ run_values[paste0(Balls, "-", 2)],
          TRUE ~ 0
        ),
        actual_event == "GroundBall" ~ run_values["GroundBall"],
        actual_event == "LineDrive" ~ run_values["LineDrive"],
        actual_event == "FlyBall" ~ run_values["FlyBall"],
        actual_event == "PopUp" ~ run_values["PopUp"],
        actual_event == "HitByPitch" ~ run_values["HitByPitch"],
        TRUE ~ 0
      ),
      
      swing_decision_rv = actual_rv - expected_rv
    ) %>%
    select(-next_count_strike, -next_count_ball)
  df
}

get_top3_swing_decisions <- function(df, batter_name, date_value) {
  date_only <- as.Date(date_value)
  
  d <- df %>%
    mutate(.date_only = as.Date(Date)) %>%
    filter(Batter == batter_name, .date_only == date_only)
  
  if (nrow(d) == 0) return(NULL)
  
  d <- calculate_expected_outcomes(d)
  d <- calculate_swing_decision_rv(d)
  
  if ("TaggedHitType" %in% names(d)) {
    d <- d %>% filter(is.na(TaggedHitType) | TaggedHitType != "Bunt")
  }
  
  swing_outcomes <- c("GroundBall", "LineDrive", "FlyBall", "PopUp", "Whiff", "Foul")
  take_outcomes <- c("CalledBall", "CalledStrike", "HitByPitch")
  
  d <- d %>%
    mutate(
      expected_swing = expected_event %in% swing_outcomes,
      expected_take = expected_event %in% take_outcomes,
      actual_swing = ifelse(is.na(actual_event), NA, actual_event %in% swing_outcomes),
      actual_take = ifelse(is.na(actual_event), NA, actual_event %in% take_outcomes)
    )
  
  if (all(is.na(d$actual_swing))) return(NULL)
  
  d_filtered <- d %>%
    filter(
      !is.na(actual_swing),
      !is.na(actual_take),
      (expected_swing == TRUE & actual_take == TRUE) |
        (expected_take == TRUE & actual_swing == TRUE)
    )
  
  if (nrow(d_filtered) == 0) return(NULL)
  
  d_filtered %>%
    arrange(desc(abs(swing_decision_rv))) %>%
    slice_head(n = 3) %>%
    mutate(
      rank = row_number(),
      decision_quality = case_when(
        swing_decision_rv > 0 ~ "Good",
        swing_decision_rv < 0 ~ "Bad",
        TRUE ~ "Neutral"
      )
    )
}

plot_single_trajectory <- function(pitch_row, rank_num,
                                   batter_image_path = "/Users/a13105/Downloads/BatterPerspective.png",
                                   x_limits = c(-2.05, 2.05),
                                   z_limits = c(0.35, 5.25),
                                   batter_xmax = -1.2,
                                   batter_width = 2.35,
                                   batter_ymin = -1.0,
                                   batter_ymax =  6.8,
                                   mound_center = c(0, 4),
                                   mound_width  = 1.55,
                                   mound_height = 0.18) {
  if (is.null(pitch_row) || nrow(pitch_row) == 0) {
    return(ggplot() + annotate("text", x = 0, y = 0, label = "No data available", size = 6) + theme_void())
  }
  
  zx_min <- -0.83; zx_max <- 0.83
  zy_min <-  1.60; zy_max <- 3.50
  
  need <- c("ax0","az0","ZoneTime","PlateLocSide","PlateLocHeight")
  if (any(is.na(unlist(pitch_row[need])))) {
    return(ggplot() + annotate("text", x = 0, y = 0, label = "Missing trajectory data", size = 5) + theme_void())
  }
  
  t_end <- as.numeric(pitch_row$ZoneTime)
  if (is.na(t_end) || t_end <= 0) {
    return(ggplot() + annotate("text", x = 0, y = 0, label = "Invalid ZoneTime", size = 5) + theme_void())
  }
  
  tm_rel_side <- as.numeric(pitch_row$RelSide)
  tm_rel_height <- as.numeric(pitch_row$RelHeight)
  
  if (!is.na(tm_rel_side) && !is.na(tm_rel_height)) {
    rel_coords <- trackman_to_plot(tm_rel_side, tm_rel_height)
    x0_use <- rel_coords$x
    z0_use <- rel_coords$z
  } else {
    x0_use <- ANCHOR_PLOT_X
    z0_use <- ANCHOR_PLOT_Z
  }
  
  x_end <- as.numeric(pitch_row$PlateLocSide)
  z_end <- as.numeric(pitch_row$PlateLocHeight)
  
  ax <- as.numeric(pitch_row$ax0)
  az <- as.numeric(pitch_row$az0)
  
  vx0_corr <- (x_end - x0_use - 0.5 * ax * t_end^2) / t_end
  vz0_corr <- (z_end - z0_use - 0.5 * az * t_end^2) / t_end
  
  t_seq <- seq(0, t_end, length.out = 180)
  traj_data <- data.frame(
    x = pitch_equation(ax, vx0_corr, x0_use, t_seq),
    z = pitch_equation(az, vz0_corr, z0_use, t_seq)
  )
  
  pitch_color <- pitch_colors[[as.character(pitch_row$TaggedPitchType)]]
  if (is.null(pitch_color)) pitch_color <- "#444444"
  
  plot_title <- paste0("#", rank_num, ": ", pitch_row$TaggedPitchType, " - ",
                       round(as.numeric(pitch_row$RelSpeed), 0), " mph")
  
  rv_value <- as.numeric(pitch_row$swing_decision_rv)
  rv_color <- if (is.na(rv_value) || abs(rv_value) < 0.001) "black" else if (rv_value > 0) "#00AA00" else "#DD0000"
  
  plot_subtitle <- paste0("Expected: ", pitch_row$expected_event,
                          " | Actual: ", pitch_row$actual_event,
                          " | RV: ", sprintf("%+.3f", pitch_row$swing_decision_rv))
  
  mound <- ellipse_df(mound_center[1], mound_center[2], mound_width, mound_height)
  batter_img <- safe_read_png(batter_image_path)
  
  batter_side <- as.character(pitch_row$BatterSide %||% "Right")
  is_lefty <- identical(tolower(batter_side), "left")
  
  if (!is.null(batter_img) && !is_lefty) {
    batter_img <- batter_img[, ncol(batter_img):1, , drop = FALSE]
  }
  
  if (is_lefty) {
    bxmin <- 1.2
    bxmax <- bxmin + batter_width
  } else {
    bxmax <- batter_xmax
    bxmin <- bxmax - batter_width
  }
  
  ggplot() +
    theme_void(base_family = "markazi") +
    geom_rect(aes(xmin = zx_min, xmax = zx_max, ymin = zy_min, ymax = zy_max),
              fill = NA, color = "black", linewidth = 1.15, linetype = "dashed") +
    geom_path(data = mound, aes(x = x, y = y), color = "black", linewidth = 1.15) +
    geom_path(data = traj_data, aes(x = x, y = z),
              color = pitch_color, linewidth = 2.8, alpha = 0.95) +
    geom_point(aes(x = x0_use, y = z0_use),
               fill = pitch_color, shape = 21, size = 2.4,
               color = "black", stroke = 0.9, alpha = 0.9) +
    geom_point(aes(x = x_end, y = z_end),
               fill = pitch_color, shape = 21, size = 5.2,
               color = "black", stroke = 1.4) +
    { if (!is.null(batter_img)) annotation_custom(
      rasterGrob(batter_img, interpolate = TRUE),
      xmin = bxmin, xmax = bxmax,
      ymin = batter_ymin, ymax = batter_ymax
    ) } +
    coord_equal(xlim = x_limits, ylim = z_limits, expand = FALSE, clip = "on") +
    labs(title = plot_title, subtitle = plot_subtitle) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold", color = rv_color),
      plot.subtitle = element_text(hjust = 0.5, size = 12, color = rv_color),
      plot.margin = margin(20, 8, 8, 8)
    )
}

# ==================== NEW: PA-by-PA Strike Zone Plots ====================
plot_pa_strike_zone <- function(df, batter_name, date_value, pa_number) {
  date_only <- as.Date(date_value)
  
  # Filter for batter/date
  d <- df %>%
    mutate(.date_only = as.Date(Date)) %>%
    filter(Batter == batter_name, .date_only == date_only)
  
  if (nrow(d) == 0) {
    return(ggplot() + theme_void() +
             annotate("text", x = 0, y = 0, label = "No data", size = 3))
  }
  
  # Assign game PA id and filter to one PA
  d <- d %>%
    group_by(Inning, PAofInning) %>%
    mutate(game_pa = cur_group_id()) %>%
    ungroup() %>%
    filter(game_pa == pa_number)
  
  if (nrow(d) == 0) {
    return(ggplot() + theme_void())
  }
  
  # Title text (PA#, inning, initial count snapshot from first pitch row)
  pa_info <- d %>%
    slice(1) %>%
    mutate(pa_desc = paste0("PA ", pa_number, " - Inn ", Inning)) %>%
    pull(pa_desc)
  
  # Build trajectories (light lines) when possible
  trajectory_list <- list()
  
  need <- c("ax0", "az0", "ZoneTime", "PlateLocSide", "PlateLocHeight",
            "RelSide", "RelHeight")
  
  for (i in seq_len(nrow(d))) {
    pitch_row <- d[i, ]
    
    if (all(need %in% names(pitch_row)) && all(!is.na(unlist(pitch_row[need])))) {
      t_end <- as.numeric(pitch_row$ZoneTime)
      if (is.finite(t_end) && t_end > 0) {
        
        tm_rel_side <- as.numeric(pitch_row$RelSide)
        tm_rel_height <- as.numeric(pitch_row$RelHeight)
        
        if (!is.na(tm_rel_side) && !is.na(tm_rel_height)) {
          rel_coords <- trackman_to_plot(tm_rel_side, tm_rel_height)
          x0_use <- rel_coords$x
          z0_use <- rel_coords$z
        } else {
          x0_use <- ANCHOR_PLOT_X
          z0_use <- ANCHOR_PLOT_Z
        }
        
        x_end <- as.numeric(pitch_row$PlateLocSide)
        z_end <- as.numeric(pitch_row$PlateLocHeight)
        ax <- as.numeric(pitch_row$ax0)
        az <- as.numeric(pitch_row$az0)
        
        vx0_corr <- (x_end - x0_use - 0.5 * ax * t_end^2) / t_end
        vz0_corr <- (z_end - z0_use - 0.5 * az * t_end^2) / t_end
        
        t_seq <- seq(0, t_end, length.out = 50)
        traj <- data.frame(
          x = pitch_equation(ax, vx0_corr, x0_use, t_seq),
          z = pitch_equation(az, vz0_corr, z0_use, t_seq),
          pitch_num = as.numeric(pitch_row$PitchofPA),
          pitch_type = as.character(pitch_row$TaggedPitchType)
        )
        
        trajectory_list[[length(trajectory_list) + 1]] <- traj
      }
    }
  }
  
  all_trajs <- if (length(trajectory_list) > 0) bind_rows(trajectory_list) else
    data.frame(x = numeric(0), z = numeric(0), pitch_num = numeric(0), pitch_type = character(0))
  
  # Endpoints with pitch numbers
  end_points <- d %>%
    mutate(
      pitch_num = as.numeric(PitchofPA),
      pitch_type = as.character(TaggedPitchType),
      x_end = as.numeric(PlateLocSide),
      z_end = as.numeric(PlateLocHeight)
    ) %>%
    filter(!is.na(x_end), !is.na(z_end))
  
  # Strike zone boundaries
  zx_min <- -0.83; zx_max <- 0.83
  zy_min <-  1.60; zy_max <- 3.50
  
  # Plot limits
  xlim_low <- -2.5; xlim_high <- 2.5
  ylim_low <- 0;    ylim_high <- 5
  
  p <- ggplot() +
    theme_void(base_family = "markazi") +
    geom_rect(aes(xmin = zx_min, xmax = zx_max, ymin = zy_min, ymax = zy_max),
              fill = NA, color = "black", linewidth = 0.8) +
    annotate("segment", x = -0.83, xend = 0.83, y = 0.5, yend = 0.5,
             colour = "black", linewidth = 0.5) +
    annotate("segment", x = -0.83, xend = -0.83, y = 0.5, yend = 0.2,
             colour = "black", linewidth = 0.5) +
    annotate("segment", x = 0.83, xend = 0.83, y = 0.5, yend = 0.2,
             colour = "black", linewidth = 0.5) +
    annotate("segment", x = 0.83, xend = 0, y = 0.2, yend = 0,
             colour = "black", linewidth = 0.5) +
    annotate("segment", x = -0.83, xend = 0, y = 0.2, yend = 0,
             colour = "black", linewidth = 0.5) +
    coord_equal(xlim = c(xlim_low, xlim_high), ylim = c(ylim_low, ylim_high)) +
    labs(title = pa_info) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 9, face = "bold"),
      plot.margin = margin(2, 2, 2, 2)
    )
  
  if (nrow(all_trajs) > 0) {
    p <- p + geom_path(
      data = all_trajs,
      aes(x = x, y = z, group = pitch_num, color = pitch_type),
      alpha = 0.4, linewidth = 0.8
    )
  }
  
  if (nrow(end_points) > 0) {
    p <- p +
      geom_point(
        data = end_points,
        aes(x = x_end, y = z_end, fill = pitch_type),
        shape = 21, size = 3, color = "black", stroke = 0.5
      ) +
      geom_text(
        data = end_points,
        aes(x = x_end, y = z_end, label = pitch_num),
        size = 2.5, fontface = "bold", color = "white"
      ) +
      scale_fill_manual(values = pitch_colors, guide = "none") +
      scale_color_manual(values = pitch_colors, guide = "none")
  }
  
  p
}

make_pitch_table <- function(df, batter_name, date_value, top3_data = NULL) {
  date_only <- as.Date(date_value)
  
  d <- df %>%
    mutate(.date_only = as.Date(Date)) %>%
    filter(Batter == batter_name, .date_only == date_only) %>%
    arrange(Inning, PAofInning, PitchofPA) %>%
    # Remove any duplicate rows first
    distinct(Inning, PAofInning, PitchofPA, .keep_all = TRUE)
  
  if (nrow(d) == 0) return(gt(data.frame(Message = "No data available")))
  
  # Create game_pa BEFORE joining with top3_data
  d <- d %>%
    group_by(Inning, PAofInning) %>%
    mutate(game_pa = cur_group_id()) %>%
    ungroup()
  
  if (!is.null(top3_data) && nrow(top3_data) > 0) {
    # Create match_id for main data
    d <- d %>% mutate(match_id = paste(Inning, PAofInning, PitchofPA, sep = "_"))
    
    # Create match_id for top3 and ensure it's unique
    top3_lookup <- top3_data %>% 
      mutate(match_id = paste(Inning, PAofInning, PitchofPA, sep = "_")) %>%
      select(match_id, rank) %>%
      distinct(match_id, .keep_all = TRUE)  # Keep only first occurrence of each match_id
    
    d <- d %>%
      left_join(top3_lookup, by = "match_id", relationship = "many-to-one") %>%
      mutate(`Key Swing Decision` = ifelse(!is.na(rank), paste0("★ #", rank), "")) %>%
      select(-match_id, -rank)
  } else {
    d <- d %>% mutate(`Key Swing Decision` = "")
  }
  
  table_data <- d %>%
    mutate(
      PA = game_pa,
      `Pitch Call` = PitchCall,
      `Play Result` = PlayResult %||% "",
      `Pitch Type` = TaggedPitchType,
      Pitcher = Pitcher,
      `Pitcher Throws` = PitcherThrows %||% "",
      Inning = Inning,
      Outs = Outs %||% NA,
      Count = paste0(Balls, "-", Strikes),
      Velo = round(RelSpeed %||% NA, 0),
      EV = round(ExitSpeed %||% NA, 0),
      LA = round(Angle %||% NA, 0)
    ) %>%
    select(`Key Swing Decision`, PA, `Pitch Call`, `Play Result`, `Pitch Type`, Pitcher,
           `Pitcher Throws`, Inning, Outs, Count, Velo, EV, LA)
  
  gt(table_data) %>%
    tab_header(
      title = md("**Pitch by Pitch Breakdown**"),
      subtitle = paste("All pitches faced on", format(date_only, "%B %d, %Y"),
                       "| ★ = Top 3 Key Swing Decisions (see trajectory plots above)")
    ) %>%
    cols_align("center") %>%
    sub_missing(everything(), missing_text = "") %>%
    tab_style(
      style = cell_borders(sides = "all", color = "black", weight = px(1)),
      locations = cells_body()
    ) %>%
    tab_style(
      style = list(cell_fill(color = "gray90"), cell_text(weight = "bold")),
      locations = cells_column_labels()
    ) %>%
    tab_style(
      style = list(cell_fill(color = "#ffffcc"), cell_text(weight = "bold", color = "#ff6600")),
      locations = cells_body(columns = `Key Swing Decision`, rows = `Key Swing Decision` != "")
    ) %>%
    tab_options(
      table.font.names = "Markazi Text",
      table.font.size = px(13),
      table.width = pct(100),
      heading.title.font.size = px(18),
      heading.subtitle.font.size = px(12),
      data_row.padding = px(4),
      table.border.top.width = px(2),
      table.border.top.color = "black",
      table.border.bottom.width = px(2),
      table.border.bottom.color = "black"
    ) %>%
    opt_row_striping()
}


# ==================== MODIFIED PDF GENERATION FUNCTION ====================
# Includes: Top row PA strike zones + middle Top3 trajectories + bottom table
generate_hitter_reports_pdf <- function(df, output_filename = "Hitter_Reports.pdf") {
  # Get most recent date
  most_recent_date <- max(as.Date(df$Date), na.rm = TRUE)
  
  # Filter for most recent date and get unique batters
  df_recent <- df %>%
    mutate(.date_only = as.Date(Date)) %>%
    filter(.date_only == most_recent_date, BatterTeam == "CAL_FUL")
  
  batters <- unique(df_recent$Batter)
  
  if (length(batters) == 0) {
    message("No batters found for the most recent date: ", most_recent_date)
    return(NULL)
  }
  
  cat("Generating PDF for", length(batters), "batters on", as.character(most_recent_date), "\n")
  cat("Saving to:", output_filename, "\n")
  
  temp_dir <- tempdir()
  
  # Enable showtext for PDF
  showtext_opts(dpi = 300)
  showtext_auto()
  
  pdf(output_filename, width = 11, height = 8.5)
  
  for (batter in batters) {
    cat("Processing:", batter, "\n")
    
    tryCatch({
      batter_data <- df_recent %>% filter(Batter == batter)
      
      # Count PAs (unique Inning/PAofInning groups)
      num_pas <- batter_data %>%
        group_by(Inning, PAofInning) %>%
        summarise(n = n(), .groups = "drop") %>%
        nrow()
      
      cat("  Found", num_pas, "plate appearances\n")
      
      # Top 3 key decisions
      top3 <- get_top3_swing_decisions(df, batter, most_recent_date)
      
      # Game date for title
      game_date <- if ("GameDate" %in% names(batter_data)) {
        unique(batter_data$GameDate)[1]
      } else {
        format(most_recent_date, "%B %d, %Y")
      }
      
      page_title <- paste0(batter, " Hitter's Report ", game_date)
      
      grid.newpage()
      
      # Title
      grid.text(page_title, x = 0.5, y = 0.97,
                gp = gpar(fontsize = 18, fontface = "bold", fontfamily = "markazi"))
      
      # ===== TOP ROW: PA-by-PA strike zone plots (max 6 shown) =====
      if (num_pas > 0) {
        pa_plot_files <- vector("list", num_pas)
        
        for (pa_num in 1:num_pas) {
          pa_plot <- plot_pa_strike_zone(df, batter, most_recent_date, pa_num)
          pa_file <- file.path(
            temp_dir,
            paste0("pa_", gsub("[^A-Za-z0-9]", "_", batter), "_", pa_num, ".png")
          )
          ggsave(pa_file, pa_plot, width = 2.2, height = 2.2, dpi = 150, bg = "white")
          pa_plot_files[[pa_num]] <- pa_file
        }
        
        # Display first up-to-6
        pushViewport(viewport(x = 0.5, y = 0.87, width = 0.96, height = 0.16))
        
        n_cols <- min(num_pas, 6)
        pa_layout <- grid.layout(nrow = 1, ncol = n_cols, widths = unit(rep(1, n_cols), "null"))
        pushViewport(viewport(layout = pa_layout))
        
        for (i in 1:n_cols) {
          f <- pa_plot_files[[i]]
          if (!is.null(f) && file.exists(f) && file.size(f) > 0) {
            pa_img <- readPNG(f)
            pushViewport(viewport(layout.pos.col = i, layout.pos.row = 1))
            grid.raster(pa_img, interpolate = TRUE)
            popViewport()
          }
        }
        
        popViewport(2)
        
        # Cleanup
        for (f in pa_plot_files) {
          if (!is.null(f) && file.exists(f)) unlink(f)
        }
      }
      
      # ===== MIDDLE ROW: Top 3 Key Swing Decisions (moved down) =====
      has_plots <- !is.null(top3) && nrow(top3) > 0
      if (has_plots) {
        pushViewport(viewport(x = 0.5, y = 0.56, width = 0.98, height = 0.38))
        
        plot_layout <- grid.layout(nrow = 1, ncol = 3, widths = unit(c(1, 1, 1), "null"))
        pushViewport(viewport(layout = plot_layout))
        
        if (nrow(top3) >= 1) {
          pushViewport(viewport(layout.pos.col = 1, layout.pos.row = 1))
          p1 <- plot_single_trajectory(top3[1, ], 1)
          print(p1, newpage = FALSE)
          popViewport()
        }
        if (nrow(top3) >= 2) {
          pushViewport(viewport(layout.pos.col = 2, layout.pos.row = 1))
          p2 <- plot_single_trajectory(top3[2, ], 2)
          print(p2, newpage = FALSE)
          popViewport()
        }
        if (nrow(top3) >= 3) {
          pushViewport(viewport(layout.pos.col = 3, layout.pos.row = 1))
          p3 <- plot_single_trajectory(top3[3, ], 3)
          print(p3, newpage = FALSE)
          popViewport()
        }
        
        popViewport(2)
      }
      
      # ===== BOTTOM: Pitch table =====
      gt_table <- make_pitch_table(df, batter, most_recent_date, top3)
      
      temp_table_file <- file.path(
        temp_dir,
        paste0("table_", gsub("[^A-Za-z0-9]", "_", batter), ".png")
      )
      
      table_saved <- tryCatch({
        gtsave(gt_table, temp_table_file, vwidth = 1400, vheight = 550)
        file.exists(temp_table_file)
      }, error = function(e) {
        cat("Warning: gtsave failed for", batter, ":", e$message, "\n")
        FALSE
      })
      
      table_y_position <- if (has_plots) 0.18 else 0.35
      table_height     <- if (has_plots) 0.30 else 0.45
      
      if (table_saved && file.exists(temp_table_file)) {
        tryCatch({
          table_img <- readPNG(temp_table_file)
          pushViewport(viewport(x = 0.5, y = table_y_position, width = 0.95, height = table_height))
          grid.raster(table_img, interpolate = TRUE)
          popViewport()
          unlink(temp_table_file)
        }, error = function(e) {
          cat("Warning: Could not read/display table for", batter, ":", e$message, "\n")
        })
      } else {
        grid.text("Pitch by Pitch Breakdown",
                  x = 0.5, y = table_y_position + 0.10,
                  gp = gpar(fontsize = 14, fontface = "bold", fontfamily = "markazi"))
      }
      
    }, error = function(e) {
      cat("Error processing", batter, ":", e$message, "\n")
      grid.newpage()
      grid.text(paste("Error generating report for", batter),
                x = 0.5, y = 0.5,
                gp = gpar(fontsize = 14, fontfamily = "markazi"))
    })
  }
  
  dev.off()
  
  # Reset showtext for screen display
  showtext_opts(dpi = 96)
  
  cat("\n=== PDF GENERATION COMPLETE ===\n")
  cat("File saved to:", output_filename, "\n")
  cat("File exists:", file.exists(output_filename), "\n")
  if (file.exists(output_filename)) {
    cat("File size:", file.size(output_filename), "bytes\n")
  }
  
  output_filename
}

# ==================== DATASET SETUP ====================
required_cols <- c("Batter", "Date", "PlateLocSide", "PlateLocHeight", "PitchCall",
                   "TaggedPitchType", "RelSpeed", "Balls", "Strikes", "BatterTeam",
                   "VertApprAngle", "HorzApprAngle")

available_datasets <- c("CSUF25", "D1TM25")
available_datasets <- available_datasets[available_datasets %in% ls(.GlobalEnv)]
if (length(available_datasets) == 0) stop("Please load CSUF25 (or D1TM25) dataset before running the app.")

default_dataset <- if ("CSUF25" %in% available_datasets) "CSUF25" else available_datasets[1]
get_dataset <- function(name) get(name, envir = .GlobalEnv)

tmp_df <- get_dataset(default_dataset)
missing_cols <- setdiff(required_cols, names(tmp_df))
if (length(missing_cols) > 0) stop(paste("Missing required columns in", default_dataset, ":", paste(missing_cols, collapse = ", ")))

# ==================== SHINY APP ====================
ui <- fluidPage(
  titlePanel("CSUF Hitter Report"),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      selectInput("dataset_name", "Dataset",
                  choices = available_datasets,
                  selected = default_dataset),
      selectizeInput("batter", "Batter",
                     choices = NULL,
                     options = list(placeholder = 'Search batter...')),
      uiOutput("date_ui"),
      hr(),
      actionButton("generate_pdf", "Generate PDF for All Hitters (Most Recent Date)",
                   class = "btn-primary", width = "100%"),
      br(), br(),
      textOutput("working_dir_display"),
      br(),
      p("Top row: Strike zones by plate appearance with pitch trajectories and numbers.",
        style = "font-size: 12px; color: #666;"),
      p("Middle row: Top 3 pitches ranked by absolute run value impact of swing decision.",
        style = "font-size: 12px; color: #666;"),
      p("★ markers in table show which pitches are plotted in middle row.",
        style = "font-size: 12px; color: #666;")
    ),
    mainPanel(
      width = 9,
      
      # NEW: PA-by-PA Strike Zones at top
      fluidRow(
        column(
          12,
          h4("Plate Appearances", style = "text-align: center; margin-bottom: 10px;"),
          uiOutput("pa_zones_ui")
        )
      ),
      hr(),
      
      # Top 3 Key Swing Decisions
      fluidRow(
        column(4, plotOutput("traj_plot_1", height = "300px", width = "400px")),
        column(4, plotOutput("traj_plot_2", height = "300px", width = "400px")),
        column(4, plotOutput("traj_plot_3", height = "300px", width = "400px"))
      ),
      br(),
      
      # Pitch table
      gt_output("pitch_table")
    )
  )
)

server <- function(input, output, session) {
  showtext_auto(TRUE)
  
  # Display working directory
  output$working_dir_display <- renderText({
    paste("PDFs will be saved to:", getwd())
  })
  
  active_df <- reactive({
    req(input$dataset_name)
    df <- get_dataset(input$dataset_name)
    
    if ("PlateLocSide" %in% names(df)) {
      df <- df %>% mutate(PlateLocSide = mirror_plate_side(PlateLocSide))
    }
    
    df <- add_LI_features(df)
    
    miss <- setdiff(required_cols, names(df))
    validate(need(length(miss) == 0, paste("Dataset missing columns:", paste(miss, collapse = ", "))))
    
    df %>% filter(BatterTeam == "CAL_FUL")
  })
  
  observeEvent(active_df(), {
    df <- active_df()
    batters_all <- sort(unique(as.character(df$Batter)))
    updateSelectizeInput(session, "batter", choices = batters_all, selected = NULL, server = TRUE)
  }, ignoreInit = FALSE)
  
  dates_for_batter <- function(df, b) {
    sort(unique(as.Date(df$Date[df$Batter == b])))
  }
  
  output$date_ui <- renderUI({
    req(input$batter)
    df <- active_df()
    ds <- dates_for_batter(df, input$batter)
    selectInput("date", "Date",
                choices = as.character(ds),
                selected = if (length(ds)) as.character(max(ds)) else NULL)
  })
  
  # NEW: number of plate appearances for selected batter/date
  num_pas <- reactive({
    req(input$batter, input$date)
    df <- active_df()
    date_only <- as.Date(input$date)
    
    d <- df %>%
      mutate(.date_only = as.Date(Date)) %>%
      filter(Batter == input$batter, .date_only == date_only)
    
    if (nrow(d) == 0) return(0)
    
    d %>%
      group_by(Inning, PAofInning) %>%
      summarise(n = n(), .groups = "drop") %>%
      nrow()
  })
  
  # NEW: dynamic UI for PA zone plots
  output$pa_zones_ui <- renderUI({
    n <- num_pas()
    if (n == 0) return(NULL)
    
    # Cap at 6 displayed to keep layout clean (matches PDF behavior)
    n_show <- min(n, 6)
    
    plot_outputs <- lapply(1:n_show, function(i) {
      plotOutput(paste0("pa_zone_", i), height = "180px", width = "180px")
    })
    
    fluidRow(
      lapply(1:n_show, function(i) {
        column(width = 2, plot_outputs[[i]])
      })
    )
  })
  
  # NEW: render each PA zone plot (first up-to-6)
  observe({
    n <- num_pas()
    if (n == 0) return()
    
    df <- active_df()
    n_show <- min(n, 6)
    
    lapply(1:n_show, function(i) {
      output_name <- paste0("pa_zone_", i)
      output[[output_name]] <- renderPlot({
        req(input$batter, input$date)
        plot_pa_strike_zone(df, input$batter, input$date, i)
      })
    })
  })
  
  # PDF Generation
  observeEvent(input$generate_pdf, {
    showModal(modalDialog(
      title = "Generating PDF",
      "Please wait while the PDF is being generated...",
      footer = NULL
    ))
    
    df <- active_df()
    
    # Print diagnostic info to console
    cat("\n=== PDF GENERATION DIAGNOSTICS ===\n")
    cat("Current working directory:", getwd(), "\n")
    cat("Files in working directory before generation:\n")
    print(list.files(pattern = "^Hitter"))
    
    pdf_filename <- paste0("Hitter_Reports_", format(Sys.Date(), "%Y%m%d"), ".pdf")
    
    # Get full path to save location
    full_path <- file.path(getwd(), pdf_filename)
    cat("Target PDF path:", full_path, "\n")
    
    tryCatch({
      result <- generate_hitter_reports_pdf(df, full_path)
      
      # Check if file was created
      cat("\n=== POST-GENERATION CHECK ===\n")
      cat("Return value from function:", result, "\n")
      cat("File exists check:", file.exists(full_path), "\n")
      cat("File exists check (relative):", file.exists(pdf_filename), "\n")
      cat("Files in working directory after generation:\n")
      print(list.files(pattern = "^Hitter"))
      
      # Try to find the file anywhere
      cat("\nSearching for file in current directory:\n")
      all_pdfs <- list.files(getwd(), pattern = "\\.pdf$", full.names = TRUE)
      cat("All PDFs found:\n")
      print(all_pdfs)
      
      removeModal()
      
      if (file.exists(full_path)) {
        showModal(modalDialog(
          title = "Success",
          HTML(paste0("PDF generated successfully!<br><br>",
                      "<strong>Filename:</strong> ", pdf_filename, "<br>",
                      "<strong>Location:</strong> ", getwd(), "<br><br>",
                      "<strong>Full path:</strong><br>", full_path, "<br><br>",
                      "<strong>File size:</strong> ",
                      format(file.size(full_path), big.mark = ","), " bytes")),
          easyClose = TRUE,
          footer = modalButton("Close")
        ))
      } else {
        showModal(modalDialog(
          title = "Warning",
          HTML(paste0("PDF generation completed but file not found at expected location.<br><br>",
                      "<strong>Expected:</strong> ", full_path, "<br><br>",
                      "Please check console output for actual file location.")),
          easyClose = TRUE,
          footer = modalButton("Close")
        ))
      }
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Failed to generate PDF:", e$message),
        easyClose = TRUE,
        footer = modalButton("Close")
      ))
    })
  })
  
  top3_data <- reactive({
    req(input$batter, input$date)
    get_top3_swing_decisions(active_df(), input$batter, input$date)
  })
  
  output$traj_plot_1 <- renderPlot({
    req(top3_data())
    top3 <- top3_data()
    if (nrow(top3) >= 1) plot_single_trajectory(top3[1, ], 1) else ggplot() + theme_void()
  })
  
  output$traj_plot_2 <- renderPlot({
    req(top3_data())
    top3 <- top3_data()
    if (nrow(top3) >= 2) plot_single_trajectory(top3[2, ], 2) else ggplot() + theme_void()
  })
  
  output$traj_plot_3 <- renderPlot({
    req(top3_data())
    top3 <- top3_data()
    if (nrow(top3) >= 3) plot_single_trajectory(top3[3, ], 3) else ggplot() + theme_void()
  })
  
  output$pitch_table <- render_gt({
    req(input$batter, input$date)
    make_pitch_table(active_df(), input$batter, input$date, top3_data())
  })
}

shinyApp(ui, server)
