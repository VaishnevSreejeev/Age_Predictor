<?php
session_start();
$error_message = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $conn = new mysqli("localhost", "root", "", "age_estimator_db");

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $email = $conn->real_escape_string($_POST['email']);
    $password = $_POST['password'];

    $sql = "SELECT * FROM users WHERE email='$email'";
    $result = $conn->query($sql);

    if ($result->num_rows == 1) {
        $user = $result->fetch_assoc();
        if (password_verify($password, $user['password'])) {
            $_SESSION['username'] = $user['username'];
            header("Location: dashboard.html");
            exit();
        } else {
            $error_message = "❌ Invalid password!";
        }
    } else {
        $error_message = "❌ No account found with that email!";
    }

    $conn->close();
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login - Age Estimator</title>
    <link rel="stylesheet" href="login.css">
</head>
<body class="login-body">
    <div class="login-container">
        <h1 class="login-title">Login</h1>

        <?php if (!empty($error_message)) { ?>
            <p class="error-msg"><?= $error_message ?></p>
        <?php } ?>

        <form class="login-form" method="POST" action="index.php">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>

        <p class="signup-text">
            Don’t have an account? <a href="signup.php">Sign Up</a>
        </p>
    </div>
</body>
</html>
