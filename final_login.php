<?php
session_start();

// Database configuration
define('DB_SERVER', 'localhost');
define('DB_USERNAME', 'root');
define('DB_PASSWORD', '');
define('DB_NAME', 'ps');

// Try connecting to the Database
$conn = mysqli_connect(DB_SERVER, DB_USERNAME, DB_PASSWORD, DB_NAME);

// Check the connection
if (!$conn) {
    die('Error: Cannot connect');
}

// Login
if (isset($_SESSION['email'])) {
    header("location: welcome.php");
    exit;
}

$email = $password = $confirm_password = "";
$email_err = $password_err = $confirm_password_err = $login_err = $err = "";

// Google Sign-In
$client = new Google_Client();
$client->setClientId('YOUR_CLIENT_ID');
$client->setClientSecret('YOUR_CLIENT_SECRET');
$client->setRedirectUri('YOUR_CALLBACK_URL');
$client->addScope("email");

if (isset($_GET['code'])) {
    $token = $client->fetchAccessTokenWithAuthCode($_GET['code']);
    $client->setAccessToken($token);

    $oauth = new Google_Service_Oauth2($client);
    $userInfo = $oauth->userinfo->get();

    // You can use $userInfo to get user details
    $email = $userInfo->getEmail();

    // Perform your login logic here using $email

    // Redirect to your welcome page after successful login
    header("location: welcome.php");
    exit;
} else {
    $authUrl = $client->createAuthUrl();
}

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    if (isset($_POST['login'])) {
        // Login process
        if (empty(trim($_POST['email'])) || empty(trim($_POST['password']))) {
            $login_err = "Please enter email address and password";
        } else {
            $email = trim($_POST['email']);
            $password = trim($_POST['password']);

            $sql = "SELECT id, email, password FROM users WHERE email = ?";
            $stmt = mysqli_prepare($conn, $sql);
            mysqli_stmt_bind_param($stmt, "s", $param_email);
            $param_email = $email;

            if (mysqli_stmt_execute($stmt)) {
                mysqli_stmt_store_result($stmt);

                if (mysqli_stmt_num_rows($stmt) == 1) {
                    mysqli_stmt_bind_result($stmt, $id, $email, $hashed_password);
                    if (mysqli_stmt_fetch($stmt)) {
                        if (password_verify($password, $hashed_password)) {
                            // Successful login
                            $_SESSION["email"] = $email;
                            $_SESSION["id"] = $id;
                            $_SESSION["loggedin"] = true;
                            header("location: welcome.php");
                            exit;
                        } else {
                            $login_err = "Invalid password";
                        }
                    }
                } else {
                    $login_err = "Invalid email";
                }
            } else {
                echo "Something went wrong...please try again later!";
            }

            mysqli_stmt_close($stmt);
        }
    } elseif (isset($_POST['register'])) {
        // Registration process
        if (empty(trim($_POST["email"]))) {
            $email_err = "Email Address cannot be blank";
        } else {
            $sql = "SELECT id FROM users WHERE email = ?";
            $stmt = mysqli_prepare($conn, $sql);
            if ($stmt) {
                mysqli_stmt_bind_param($stmt, "s", $param_email);

                $param_email = trim($_POST['email']);

                if (mysqli_stmt_execute($stmt)) {
                    mysqli_stmt_store_result($stmt);
                    if (mysqli_stmt_num_rows($stmt) == 1) {
                        $email_err = "This email has already been registered";
                    } else {
                        $email = trim($_POST['email']);
                    }
                } else {
                    echo "Something went wrong...please try again later!";
                }
            }
        }

        mysqli_stmt_close($stmt);

        if (empty(trim($_POST['password']))) {
            $password_err = "Password cannot be blank";
        } elseif (strlen(trim($_POST['password'])) < 5) {
            $password_err = "Password cannot be less than 5 characters";
        } else {
            $password = trim($_POST['password']);
        }

        if (empty($email_err) && empty($password_err) && empty($confirm_password_err)) {
            $sql = "INSERT INTO users (email, password) VALUES (?, ?)";
            $stmt = mysqli_prepare($conn, $sql);
            if ($stmt) {
                mysqli_stmt_bind_param($stmt, "ss", $param_email, $param_password);

                $param_email = $email;
                $param_password = password_hash($password, PASSWORD_DEFAULT);

                if (mysqli_stmt_execute($stmt)) {
                    header("location: login.php");
                    exit();
                } else {
                    echo "Something went wrong...cannot redirect...please try again later!";
                }
            }
            mysqli_stmt_close($stmt);
        }
    }
} elseif (isset($_POST['logout'])) {
    // Logout
    $_SESSION = array();
    session_destroy();
    header("location: login.php");
    exit();
}

mysqli_close($conn);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Login and Register</title>
    <link rel="stylesheet" href="login.css"/>
    <link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet"/>
    <script src="https://kit.fontawesome.com/bac43738cb.js" crossorigin="anonymous"></script>
</head>
<body>
<header class="header">
    <a href="#" class="cyber">Cyber</a>
    <nav class="navbar">
        <a href="web.html">Home</a>
        <a href="#">About</a>
        <a href="#">Services</a>
        <a href="#">Contact</a>
        <button class="btnLogin-popup">Login</button>
    </nav>
</header>

<!-- <section class="section"> -->
<div class="wrapper">
    <div class="logreg-box">
        <!-- login form -->
        <div class="form-box login">
            <div class="logreg-title">
                <h2>Login</h2>
                <br/>
                <p>Please login to use the platform</p>
            </div>
            <form action="final_login.php" method="post">
                <div class="input-box">
                    <span class="icon"><i class="bx bxs-envelope"></i></span>
                    <input type="email" name="email" required/>
                    <label>Email</label>
                </div>

                <div class="input-box">
                    <span class="icon"><i class="fa-solid fa-eye Eyecon"></i></span>
                    <input type="password" name="password" required id="MyInput"/>
                    <label>Password</label>
                </div>

                <div class="remember-forgot">
                    <label><input type="checkbox"/>Remember me</label>
                    <a href="#">Forgot password?</a>
                </div>

                <button type="submit" class="btn" name="login">Login</button>
                <div class="logreg-link">
                    <p>
                        Don't have an account?
                        <a href="#" class="register-link">Register</a>
                    </p>
                </div>
            </form>
        </div>

        <!-- register form -->
        <div class="form-box register">
            <div class="logreg-title">
                <h2>Registration</h2>
                <br/>
                <p>Please provide the following to verify your identity</p>
            </div>
            <form action="final_login.php" method="post">
                <div class="input-box">
                    <span class="icon"><i class="bx bxs-envelope"></i></span>
                    <input type="email" name="email" required/>
                    <label>Email</label>
                </div>

                <div class="input-box">
                    <span class="icon"><i class="fa-solid fa-eye Eyecon"></i></span>
                    <input type="password" name="password" required id="MyInput"/>
                    <label>Password</label>
                </div>

                <div class="input-box">
                    <span class="icon"><i class="fa-solid fa-eye Eyecon"></i></span>
                    <input type="password" name="confirm_password" required id="ConfInput"/>
                    <label>Confirm Password</label>
                </div>

                <!-- <div class="remember-forgot">
                    <label><input type="checkbox" required> I agree to the terms & conditions</label>
                </div> -->

                <button type="submit" class="btn" name="register">Register</button>
                <div class="logreg-link">
                    <p>
                        Already have an account?
                        <a href="#" class="login-link">Login</a>
                    </p>
                </div>
            </form>
        </div>
    </div>
    <div class="media-options">
        <a href="#">
            <i class="bx bxl-google"></i>
            <span>Login with Google</span>
        </a>
        <!-- <a href="#">
            <i class='bx bxl-facebook' ></i>
            <span>Login with Facebook</span>
        </a> -->
    </div>
</div>
<!-- </section> -->

<script src="login.js"></script>
</body>
</html>
