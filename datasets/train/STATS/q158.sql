select  count(*) from comments as c,  		posts as p,          users as u where c.UserId = u.Id 	and u.Id = p.OwnerUserId  AND c.Score=0  AND p.Score>=-1  AND p.Score<=14  AND p.ViewCount<=3944  AND p.AnswerCount>=0  AND p.CommentCount>=0  AND u.Reputation=1  AND u.DownVotes=0  AND u.UpVotes<=39  AND u.CreationDate<='2014-08-19 20:55:25'::timestamp;