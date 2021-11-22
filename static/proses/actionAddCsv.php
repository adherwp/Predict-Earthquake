<?php
$code = $_FILES['file']['error'];
if ($code === 0) {     

    $destination_path = getcwd();

    $error = "";
    $nama_folder = "upload";
    $tmp = $_FILES['file']['tmp_name'];
    $nama_file = $_FILES['file']['name'];
    $path = $destination_path . "\\..\\..\\$nama_folder\\$nama_file";

    if (file_exists($path)) {
        $error = urldecode("File dengan nama yang sama sudah tersimpan, coba lagi");
        header("Location:../../templates/training.php?error=$error");
    }

    $tipe_file = array('application/vnd.ms-excel','text/plain','text/csv','text/tsv');
    if (!in_array($_FILES['file']['type'], $tipe_file)) {
        $error = urldecode("Cek Kembali Ekstensi File Anda (*jpeg, *jpg, *gif, *png)");
        header("Location:../../templates/training.php?error=$error");
    }

    if(move_uploaded_file($tmp, $path)) {
        header("Location:../../templates/training.php");
    }
    else{
        $error = urldecode("Data tidak berhasil ditambahakan");
        header("Location:../../templates/training.php?error=$error");
    }
} 
else {
    $error = urldecode("csv tidak berhasil terupload");
    header("Location:../../templates/training.php?error=$error");
}

mysqli_close($con);
?>