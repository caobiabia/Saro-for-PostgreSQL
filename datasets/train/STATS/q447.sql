select  count(*) from badges as b, 		users as u where b.UserId= u.Id  AND b.Date<='2014-09-12 03:29:35'::timestamp  AND u.Reputation>=1  AND u.UpVotes>=0  AND u.UpVotes<=154  AND u.CreationDate<='2014-08-16 06:03:32'::timestamp;