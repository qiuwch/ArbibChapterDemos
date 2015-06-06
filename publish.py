#!env python
# script to sync publish this folder to a web server
import time
import glob
import os
import boto
import boto.s3
import os.path
import sys

class UpdateManager:
    def __init__(self):
        self.datetimeLog = {}
        self.logFilename = '.log'
        self.timeFormat = '%Y-%m-%dT%H:%M:%SZ'
        self.load()

    def load(self):
        # parse stored log
        import os
        if os.path.exists(self.logFilename):
            f = open(self.logFilename, 'r')
            lines = f.readlines()
            for line in lines:
                [a, b] = line.strip().split('\t')
                self.datetimeLog[a] = time.strptime(b, self.timeFormat)

    def save(self):
        # save updated log
        f = open(self.logFilename, 'w')
        for filename in self.datetimeLog.keys():
            f.write(filename + '\t' + time.strftime(self.timeFormat, self.datetimeLog[filename]) + '\n')
        f.close()

    def needUpdate(self, filename):
        updateTime = time.strptime(time.ctime(os.path.getmtime(filename)))
        if self.datetimeLog.get(filename) and updateTime <= self.datetimeLog[filename]:
            return False # No need to update
        else:
            return True

    def update(self, filename):
        updateTime = time.strptime(time.ctime(os.path.getmtime(filename)))
        self.datetimeLog[filename] = updateTime


class S3Uploader:
    ''' Script to upload file to s3 '''
    def __init__(self):
        # Fill in info on data to upload
        # destination bucket name
        self.bucket_name = 'arbib-chapter-demos'
        # source directory
        sourceDir = '.'
        # destination directory name (on s3)
        destDir = ''
         
        #max size in bytes before uploading in parts. between 1 and 5 GB recommended
        self.MAX_SIZE = 20 * 1000 * 1000
        #size of parts when uploading in parts
        self.PART_SIZE = 6 * 1000 * 1000
         
        #conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET)
        conn = boto.connect_s3(host='s3-us-west-2.amazonaws.com') # a bug of boto requires to set the host, https://github.com/boto/boto/issues/621
         
        #bucket = conn.create_bucket(bucket_name,
                #location=boto.s3.connection.Location.DEFAULT)
        self.bucket = conn.get_bucket(self.bucket_name)

    def uploadFile(self, filename, destpath):
        sourcepath = filename
        print 'Uploading %s --> %s to Amazon S3 bucket %s' % \
               (sourcepath, destpath, self.bucket_name)
     
        filesize = os.path.getsize(sourcepath)
        if filesize > self.MAX_SIZE:
            print "multipart upload"
            mp = self.bucket.initiate_multipart_upload(destpath)
            fp = open(sourcepath,'rb')
            fp_num = 0
            while (fp.tell() < filesize):
                fp_num += 1
                print "uploading part %i" %fp_num
                mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10, size=PART_SIZE)

            mp.complete_upload()
     
        else:
            print "singlepart upload"
            k = boto.s3.key.Key(self.bucket)
            k.key = destpath
            k.set_contents_from_filename(sourcepath,
                    cb=self.percent_cb, num_cb=10)
    
    ''' Not implemented yet    
    def uploadFolder(self, folder):
        uploadFileNames = []
        for (root, dirs, files) in os.walk(sourceDir):
            #print filename
            filenames = [root + '/' + item for item in files]
            #uploadFileNames.extend(filename)
            uploadFileNames.extend(filenames)
            #break

        uploadFileNames = [a for a in uploadFileNames if not exclude(a)]
        destpath = os.path.join(filename.split('/')[-1])
    '''

    # exclude image file
    def exclude(self, filename):
        dirs = filename.split('/')[:-1]
        excludeFolder = ['.git', 'trash', 'reference', '.ipynb_checkpoints']
        for folder in dirs:
            if folder in excludeFolder:
                return True

        for ext in ['.py', '.git', '.swp', '.zip', '.DS_Store', '.log', '.gitignore']:
            if filename.endswith(ext):
                return True
        else:
            return False

    def percent_cb(self, complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()


def exportHTML(filename, htmlFilename):
    import glob
    import os
    # import IPython.nbconvert
    # from IPython.nbformat import current as nbformat
    from IPython.nbconvert import HTMLExporter
    from IPython.config import Config

    # with open(filename, 'r') as f:
    #     notebook = nbformat.read(f, 'ipynb')

    exportHtml = HTMLExporter(config=None, template_file='./src/data/other/full.tpl')
    # exportHtml = HTMLExporter()
    
    print 'Convert from %s to %s' % (filename, htmlFilename)
    (body,resources) = exportHtml.from_file(filename)

    with open(htmlFilename, 'w') as f:
        f.write(body.encode('utf-8'))

def exportPY(filename, pyFilename):
    from IPython.nbconvert import PythonExporter
    pythonExporter = PythonExporter()
    (body, resources) = pythonExporter.from_file(filename)
    with open(pyFilename, 'w') as f:
        f.write(body.encode('utf-8'))


def publish():
    s3uploader = S3Uploader()
    manager = UpdateManager()
    files = glob.glob('./src/ipynb/*.ipynb')

    # Convert ipynb to html
    for filename in files:
        pyFilename = './src/py/' + filename.split('/')[-1].replace('.ipynb', '.py')
        exportPY(filename, pyFilename)

        if manager.needUpdate(filename):
            htmlFilename = './src/html/' + filename.split('/')[-1].replace('.ipynb', '.html')
            destpath = 'html/' + htmlFilename.split('/')[-1]

            exportHTML(filename, htmlFilename)
            s3uploader.uploadFile(htmlFilename, destpath)
            s3uploader.uploadFile(filename, 'ipynb/' + filename.split('/')[-1])            

            manager.update(filename)

    # Upload data files
    datafiles = []
    for (root, dirs, files) in os.walk('src/data'):
        filenames = [root + '/' + item for item in files]
        datafiles.extend(filenames)

    for filename in datafiles:
        if (not s3uploader.exclude(filename)) and manager.needUpdate(filename):
            destpath = filename.replace('src/', '')
            s3uploader.uploadFile(filename, destpath)
            manager.update(filename)

    s3uploader.uploadFile('./src/index.html', 'index.html')

    manager.save()

publish()
