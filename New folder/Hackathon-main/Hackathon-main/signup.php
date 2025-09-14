<?php
$success_message = "";
$error_message = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $conn = new mysqli("localhost", "root", "", "age_estimator_db");

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $username = $conn->real_escape_string($_POST['username']);
    $email = $conn->real_escape_string($_POST['email']);
    $password = password_hash($_POST['password'], PASSWORD_DEFAULT);

    $sql = "INSERT INTO users (username, email, password) VALUES ('$username', '$email', '$password')";

    if ($conn->query($sql) === TRUE) {
        $success_message = "✅ Account successfully created!";
        $_POST = array(); // clear form fields
    } else {
        $error_message = "⚠️ Error: " . $conn->error;
    }

    $conn->close();
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sign Up - Age Estimator</title>
    <link rel="stylesheet" href="signup.css">
</head>
<body class="signup-body">
    <div class="signup-container">
        <h1 class="signup-title">Create Account</h1>

        <?php if (!empty($success_message)) { ?>
            <p class="success-msg"><?= $success_message ?></p>
        <?php } ?>
        <?php if (!empty($error_message)) { ?>
            <p class="error-msg"><?= $error_message ?></p>
        <?php } ?>

        <form class="signup-form" method="POST" action="signup.php">
            <input type="text" name="username" value="<?= $_POST['username'] ?? '' ?>" placeholder="Username" required>
            <input type="email" name="email" value="<?= $_POST['email'] ?? '' ?>" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Sign Up</button>
        </form>

        <p class="login-text">
            Already have an account? <a href="index.php">Login</a>
        </p>
    </div>
</body>
</html>
