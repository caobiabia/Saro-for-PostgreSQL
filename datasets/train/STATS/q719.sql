select  count(*) from comments as c,          badges as b,         users as u where u.Id = c.UserId 	and c.UserId = b.UserId  AND c.Score=0  AND u.Reputation>=1  AND u.Reputation<=135  AND u.Views>=0  AND u.UpVotes<=14  AND u.CreationDate<='2014-09-02 18:10:48'::timestamp;